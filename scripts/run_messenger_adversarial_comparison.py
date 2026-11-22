"""Three loss conditions, position-selected checkpoints; no clean-feature head path."""
import argparse
from dataclasses import asdict
import fcntl
import gc
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_messenger_position_primary as primary
from scripts.run_messenger_multiaxis_convergence import Plateau, save, signature, set_lr
from src.models.messenger_feature_discriminator import (
    FeaturePatchDiscriminator, paired_candidates, receptive_masks, spatial_mean,
    discriminator_objective, generator_objectives)
base, old = primary.base, primary.old
OUT = ROOT/'outputs/diagnostics/messenger_adversarial_comparison_20261119'
ARMS = ('baseline', 'adversarial', 'adversarial_fm')
PLAN = dict(primary.PLAN, arms=ARMS, architecture='regression scale1/2/4 width96; no branch auxiliary',
    adversarial_weight=.001, feature_matching_weight=.01,
    D='conditional SN CNN 771->128(1x1)->128(3x3)->192(3x3,s2)->192(3x3)->1(1x1)',
    D_seed=43, D_lr=1e-4, D_weight_decay=0., D_updates_per_R=1,
    D_inputs='candidate, occluded X, artificial mask, xy; both candidates share outside X',
    D_support='per-image mean over receptive fields intersecting mask; three FM levels',
    D_normalization='fixed per-channel clean fit mean/std (floor .01); no batch or sample norm',
    R_convergence='shared stationary dev position and position+.01MSE+.01cosine; min40, lr floor1e-6, 8/15 patience; cap600 not convergence',
    adversarial_convergence='adversarial/FM/D curves recorded but moving D objectives do not select or extend common position plateau',
    checkpoint='best_R includes matching D; last_R includes D optimizer, buffers and RNG',
    coefficient_choice='predeclared small auxiliary weights, not tuned or claimed optimal; initial R-gradient ratios logged',
    scope='same reused small train-video split, seed42; not global val/test or production',
    clean_head=False, retention='all 12 best/last R/g files retained through numerical audit; no automatic deletion')


def read(path):
    return json.loads(Path(path).read_text())


def new_recovery():
    return old.make_net('det_position', None)


def new_discriminator(stats):
    return FeaturePatchDiscriminator(*stats).cuda()


@torch.no_grad()
def validate(net, discriminator, model, cache):
    result = primary.validate(net, model, cache)
    if discriminator is None:
        return result
    discriminator.eval()
    values = []
    for b in base.batches(len(cache['syn_dev']), 4):
        j = cache['syn_dev'][b]
        maps, z, c, _ = base.syn_batch(cache, j)
        r = primary.generate(net, maps, j)
        real, fake = paired_candidates(maps[:, 2], r, z, c)
        rl, _ = discriminator(real, maps[:, 2], c)
        fl, _ = discriminator(fake, maps[:, 2], c)
        adv, fm = generator_objectives(discriminator, maps[:, 2], r, z, c)
        support = receptive_masks(c)[-1]
        values.append([float(adv), float(fm), float(discriminator_objective(rl, fl, c)),
                       float(spatial_mean((rl > 0).float(), support)),
                       float(spatial_mean((fl < 0).float(), support))])
    result.update(dict(zip(('adversarial', 'feature_matching', 'D_loss', 'D_real_accuracy', 'D_fake_accuracy'),
                           np.mean(values, axis=0).tolist())))
    return result


def train_recovery(folder, arm, model, cache, stats, identity):
    net = new_recovery()
    opt, ctrl, rng, history, best, start, done = old.initial(net, folder, 'R', identity)
    discriminator = None if arm == 'baseline' else new_discriminator(stats)
    dopt = None if discriminator is None else torch.optim.AdamW(
        discriminator.parameters(), lr=PLAN['D_lr'], weight_decay=PLAN['D_weight_decay'])
    if (folder/'last_R.pt').exists() and discriminator is not None:
        ck = torch.load(folder/'last_R.pt', map_location='cpu', weights_only=True)
        discriminator.load_state_dict(ck['discriminator'])
        dopt.load_state_dict(ck['D_optimizer'])
        torch.cuda.set_rng_state_all(ck['cuda_rng'])
    initial_hash = base.tensor_digest(net)
    if start == 0:
        base.write(folder/'initialization.json', dict(R=initial_hash,
            D=None if discriminator is None else base.tensor_digest(discriminator)))
    for epoch in range(start, 601) if not done else ():
        tick = time.monotonic()
        net.train()
        totals = []
        if epoch:
            ids = rng.permutation(cache['syn_fit'])
            for bb in base.batches(len(ids), 16):
                group = ids[bb]
                if discriminator is not None:
                    discriminator.train().requires_grad_(True)
                    dopt.zero_grad(set_to_none=True)
                    for mb in base.batches(len(group), 4):
                        j = group[mb]; maps, z, c, _ = base.syn_batch(cache, j)
                        with torch.no_grad():
                            r = primary.generate(net, maps, j)
                        real, fake = paired_candidates(maps[:, 2], r, z, c)
                        rl, _ = discriminator(real, maps[:, 2], c)
                        fl, _ = discriminator(fake, maps[:, 2], c)
                        dl = discriminator_objective(rl, fl, c)*len(j)/len(group)
                        dl.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5, error_if_nonfinite=True)
                    dopt.step(); dopt.zero_grad(set_to_none=True)
                    discriminator.eval().requires_grad_(False)
                opt.zero_grad(set_to_none=True)
                gp = cache['point'][group]
                gv = cache['state'][cache['parent'][group]] > 0
                denominators = [int((gp & gv).sum()), int((~gp & gv).sum())]
                for mb in base.batches(len(group), 4):
                    j = group[mb]; maps, z, c, e = base.syn_batch(cache, j)
                    r = primary.generate(net, maps, j)
                    xy = base.gpu(cache['xy'][cache['parent'][j]])
                    point = base.gpu(cache['point'][j]); valid = base.gpu(cache['state'][cache['parent'][j]]) > 0
                    _, parts, err = primary.endpoint_loss(model, r, maps[:, 2], z, c, e, xy, point, valid)
                    pos = sum(err[m].sum()/max(d, 1) for m, d in
                              zip((point & valid, ~point & valid), denominators))/sum(d > 0 for d in denominators)
                    loss = pos+(.01*parts['mse']+.01*parts['cosine'])*len(j)/len(group)
                    if discriminator is not None:
                        adv, fm = generator_objectives(discriminator, maps[:, 2], r, z, c)
                        extra = PLAN['adversarial_weight']*adv
                        if arm == 'adversarial_fm':
                            extra = extra+PLAN['feature_matching_weight']*fm
                        loss = loss+extra*len(j)/len(group)
                    loss.backward(); totals.append(float(loss.detach()))
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5, error_if_nonfinite=True)
                opt.step()
        dev = validate(net, discriminator, model, cache)
        dev['total_G_objective'] = dev['objective'] + (PLAN['adversarial_weight']*dev.get('adversarial', 0.)) + (
            PLAN['feature_matching_weight']*dev.get('feature_matching', 0.) if arm == 'adversarial_fm' else 0.)
        if not all(np.isfinite(v) for v in dev.values()):
            raise ValueError('Nonfinite dev diagnostic')
        if dev['position'] < best:
            best = dev['position']
            save(folder/'best_R.pt', dict(state=net.state_dict(), epoch=epoch, dev=dev, identity=identity,
                discriminator=None if discriminator is None else discriminator.state_dict()))
        done = ctrl.step(epoch, [dev['position'], dev['objective']])
        set_lr(opt, ctrl.lr)
        if dopt is not None:
            set_lr(dopt, ctrl.lr)
        row = dict(epoch=epoch, dev=dev, train=sum(totals)/48 if totals else None,
                   lr=ctrl.lr, seconds=time.monotonic()-tick, converged=done)
        history.append(row)
        base.write(folder/'R_history.json', history)
        save(folder/'last_R.pt', dict(state=net.state_dict(), optimizer=opt.state_dict(),
            controller=asdict(ctrl), rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(), epoch=epoch, best=best, identity=identity, converged=done,
            discriminator=None if discriminator is None else discriminator.state_dict(),
            D_optimizer=None if dopt is None else dopt.state_dict()))
        base.status(OUT, 'training', arm=arm, stage='R', **row)
        if done:
            break
    if not done:
        raise RuntimeError('R cap reached without position convergence')
    ck = torch.load(folder/'best_R.pt', weights_only=True)
    net.load_state_dict(ck['state']); net.eval().requires_grad_(False)
    if discriminator is not None:
        discriminator.load_state_dict(ck['discriminator']); discriminator.eval().requires_grad_(False)
        base.write(folder/'selected_D_dev.json', validate(net, discriminator, model, cache))
    base.write(folder/'R_convergence.json', dict(converged=True, selected_epoch=ck['epoch'],
        stop_epoch=history[-1]['epoch'], criterion=PLAN['R_convergence']))
    return net, discriminator


def gradient_norm(loss, net):
    gs = torch.autograd.grad(loss, tuple(net.parameters()), retain_graph=True, allow_unused=True)
    return float(sum(g.square().sum() for g in gs if g is not None).sqrt())


def technical(model, cache, stats):
    j = cache['syn_fit'][:4]; maps, z, c, e = base.syn_batch(cache, j)
    xy = base.gpu(cache['xy'][cache['parent'][j]])
    point = base.gpu(cache['point'][j]); valid = base.gpu(cache['state'][cache['parent'][j]]) > 0
    results = {}
    for arm in ARMS:
        net = new_recovery(); net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
        d = None if arm == 'baseline' else new_discriminator(stats)
        do = None if d is None else torch.optim.AdamW(d.parameters(), lr=1e-4, weight_decay=0.)
        losses = []; gradients = None; d_changed = False
        for step in range(12):
            if d is not None:
                d.train().requires_grad_(True); do.zero_grad(set_to_none=True)
                dh = base.tensor_digest(d)
                with torch.no_grad():
                    r = primary.generate(net, maps, j)
                real, fake = paired_candidates(maps[:, 2], r, z, c)
                rl, _ = d(real, maps[:, 2], c); fl, _ = d(fake, maps[:, 2], c)
                dl = discriminator_objective(rl, fl, c); dl.backward()
                torch.nn.utils.clip_grad_norm_(d.parameters(), 5, error_if_nonfinite=True)
                do.step(); do.zero_grad(set_to_none=True)
                d_changed |= base.tensor_digest(d) != dh
                d.eval().requires_grad_(False)
            opt.zero_grad(set_to_none=True)
            r = primary.generate(net, maps, j)
            loss, parts, _ = primary.endpoint_loss(model, r, maps[:, 2], z, c, e, xy, point, valid)
            if step == 0:
                gradients = dict(position=gradient_norm(parts['position'], net),
                    weighted_feature=gradient_norm(.01*parts['mse']+.01*parts['cosine'], net))
            if d is not None:
                adv, fm = generator_objectives(d, maps[:, 2], r, z, c)
                if step == 0:
                    gradients['weighted_adversarial'] = gradient_norm(PLAN['adversarial_weight']*adv, net)
                    gradients['weighted_FM'] = gradient_norm(PLAN['feature_matching_weight']*fm, net)
                loss = loss+PLAN['adversarial_weight']*adv
                if arm == 'adversarial_fm':
                    loss = loss+PLAN['feature_matching_weight']*fm
            loss.backward()
            assert all(p.grad is None for p in model.parameters())
            if d is not None:
                assert all(p.grad is None for p in d.parameters())
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5, error_if_nonfinite=True)
            opt.step(); losses.append(float(loss.detach()))
        assert all(np.isfinite(v) and v > 0 for v in gradients.values())
        assert losses[-1] < losses[0], ('technical_overfit_failed', arm, losses)
        assert d is None or d_changed
        results[arm] = dict(losses=losses, gradients=gradients,
                           D_updated=d_changed, fixed_head_no_grad=True)
        del net, opt, d, do, r, loss
        gc.collect(); torch.cuda.empty_cache()
    base.write(OUT/'technical_checks.json', results)


@torch.no_grad()
def extra_evaluation(folder, net, gate, discriminator, model, cache):
    result = {}
    for domain in ('natural', 'artificial'):
        ids = cache['natural_eval' if domain == 'natural' else 'syn_eval']
        gs = []; feature_components = []; norm_values = []
        drows = []
        for b in base.batches(len(ids), 4):
            j = ids[b]
            if domain == 'natural':
                maps = base.gpu(cache['clean'][j]); xy = base.gpu(cache['xy'][j])
            else:
                maps, z, c, e = base.syn_batch(cache, j)
                xy = base.gpu(cache['xy'][cache['parent'][j]])
            x = maps[:, 2]; r = primary.generate(net, maps, j); logits, y = gate(x, r, 1.)
            ij = ((xy*255+.5)/16).floor().long().clamp(0, 15)
            gs.append(logits.sigmoid()[:, 0][torch.arange(len(j), device='cuda')[:, None], ij[:, :, 1], ij[:, :, 0]].cpu().numpy())
            norm_values.append(torch.stack([v.norm(dim=1).mean((1, 2)) for v in (x, r, y)], 1).cpu().numpy())
            if domain == 'artificial':
                feature_components.append([float(old.region_mean((r-z).square().mean(1), c, e)),
                    float(old.region_mean(1-F.cosine_similarity(r, z, dim=1), c, e))])
                if discriminator is not None:
                    real, fake = paired_candidates(x, r, z, c)
                    rl, _ = discriminator(real, x, c); fl, _ = discriminator(fake, x, c)
                    support = receptive_masks(c)[-1]
                    drows.append([float(spatial_mean(rl, support)), float(spatial_mean(fl, support)),
                        float(spatial_mean((rl > 0).float(), support)), float(spatial_mean((fl < 0).float(), support))])
        gates = np.concatenate(gs)
        masks = {'occluded': cache['state'][ids].numpy() == 1, 'visible': cache['state'][ids].numpy() == 2} if domain == 'natural' else {
            'covered': cache['point'][ids].numpy(), 'uncovered': ~cache['point'][ids].numpy()}
        parent = ids if domain == 'natural' else cache['parent'][ids]
        np.savez_compressed(folder/f'{domain}_extras.npz', gate=gates, norms=np.concatenate(norm_values),
            contour=cache['contour'][parent], ratio=cache['ratio'][parent], state=cache['state'][parent].numpy(),
            annotation_ids=np.array(read(OUT/'manifest.json')['faces'])[parent])
        result[domain] = dict(gate={label: dict(mean=float(gates[m].mean()), median=float(np.median(gates[m]))) for label, m in masks.items()},
            mean_feature_norms_X_R_fused=np.concatenate(norm_values).mean(0).tolist())
        if feature_components:
            result[domain]['mse_cosine_region_mean'] = np.mean(feature_components, axis=0).tolist()
        if drows:
            result[domain]['D_real_fake_logits_accuracies'] = np.mean(drows, axis=0).tolist()
    base.write(folder/'extra_evaluation.json', result)
    return result


def main(resume=False, wait=False):
    if resume:
        assert OUT.is_dir()
    else:
        OUT.mkdir(exist_ok=False)
    # Process-local redirection only. Existing runners and their sources are unchanged.
    primary.OUT = OUT
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = True
    prior = read(ROOT/'outputs/diagnostics/messenger_position_primary_20261118/plan.json')
    for p, h in prior['hashes'].items():
        assert base.sha(ROOT/p) == h, p
    paths = list(prior['hashes']) + [str(Path(__file__).relative_to(ROOT)),
        'src/models/messenger_feature_discriminator.py', 'tests/test_messenger_feature_discriminator.py',
        'scripts/report_messenger_adversarial_comparison.py', 'docs/messenger_adversarial_comparison_20261119.md']
    plan = dict(**PLAN, hashes={p: base.sha(ROOT/p) for p in paths})
    identity = signature(plan)
    if resume:
        assert read(OUT/'plan.json') == json.loads(json.dumps(plan))
    else:
        base.write(OUT/'plan.json', plan)
        for p in paths:
            if p.endswith('.py'):
                dest = OUT/'source_snapshot'/p
                dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/p, dest); dest.chmod(0o444)
    with (OUT/'singleton.lock').open('a') as singleton:
        fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with (ROOT/'outputs/training_queues/pending_models_tests_then_training/queue.lock').open('a') as lease:
                while True:
                    try:
                        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if not wait:
                            raise
                        base.status(OUT, 'waiting_for_canonical_lease')
                        time.sleep(15)
                for p, h in plan['hashes'].items():
                    assert base.sha(ROOT/p) == h, p
                base.status(OUT, 'preparing')
                config = yaml.safe_load(base.CONFIG.read_text())
                model, _ = base.build_landmark_model(config, variant='query_refinement', initialize_from_detector=False)
                base.load_landmark_checkpoint(model, base.BASE, prefer_ema=True, strict=True)
                model.cuda().eval().requires_grad_(False)
                frozen = base.tensor_digest(model)
                cache = base.make_cache(OUT, model, config['data']); manifest = read(OUT/'manifest.json')
                for k in ('roles', 'faces', 'image_hashes'):
                    assert manifest[k] == read(old.OUT/'manifest.json')[k]
                base.write(OUT/'teacher_audit.json', old.teacher_cache(cache, manifest))
                fit = np.unique(cache['parent'][cache['syn_fit']]); f = cache['clean'][fit, 2].float()
                stats = (f.mean((0, 2, 3)), f.std((0, 2, 3), unbiased=False).clamp_min(.01)); del f
                base.write(OUT/'normalization.json', dict(fit=fit.tolist(), mean=stats[0].tolist(), std=stats[1].tolist()))
                if not (OUT/'technical_checks.json').exists():
                    base.status(OUT, 'technical_checks'); technical(model, cache, stats)
                results = {}
                for arm in ARMS:
                    folder = OUT/arm; folder.mkdir(exist_ok=True)
                    aid = signature(dict(plan=identity, arm=arm))
                    if (folder/'evaluation.json').exists():
                        m = read(folder/'evaluation.json'); assert m['identity'] == aid
                        for p, h in m['weights'].items():
                            assert base.sha(folder/p) == h
                    else:
                        net, discriminator = train_recovery(folder, arm, model, cache, stats, aid)
                        recovered = primary.candidates(net, cache)
                        gate = primary.train_gate(folder, arm, model, cache, recovered, aid)
                        base.status(OUT, 'evaluating', arm=arm)
                        m = primary.evaluate(folder, net, gate, model, cache)
                        m['extras'] = extra_evaluation(folder, net, gate, discriminator, model, cache)
                        m.update(identity=aid, R=read(folder/'R_convergence.json'), g=read(folder/'g_convergence.json'),
                                 weights={p.name: base.sha(p) for p in folder.glob('*.pt')})
                        base.write(folder/'evaluation.json', m)
                        del net, gate, recovered, discriminator; gc.collect(); torch.cuda.empty_cache()
                    results[arm] = m; base.write(OUT/'comparison.json', results)
                    assert base.tensor_digest(model) == frozen and all(p.grad is None for p in model.parameters())
                for p, h in plan['hashes'].items():
                    assert base.sha(ROOT/p) == h, p
                files = [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=base.sha(p)) for p in OUT.rglob('*.pt')]
                assert len(files) == 12
                base.write(OUT/'summary.json', dict(conditions=results, checkpoints=files, checkpoint_bytes=sum(p['bytes'] for p in files)))
                base.write(OUT/'training_completion.json', dict(all_converged=True, weights_retained=True,
                    clean_head=False, fixed_parent=frozen, source_verified=True))
                from scripts.report_messenger_adversarial_comparison import report
                report(OUT)
                base.status(OUT, 'completed', arms=3, all_numerical_evaluation=True, weights_retained=True)
        except Exception as error:
            base.status(OUT, 'failed', error=repr(error))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--wait', action='store_true')
    args = parser.parse_args(); main(args.resume, args.wait)
