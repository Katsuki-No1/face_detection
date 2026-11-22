"""Joint discrete composition policy, full-crop D, and separated G/D/T updates.

Deployment stays HEAD-free FACE/ellipse -> crop DINO -> Messenger -> Query.
G and D are train-only. Policy gradients estimate the hard selection gradient;
no straight-through mask/alpha or gradients into recipient/donor pixels.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.data.messenger_policy_occlusion import compose
from src.models.messenger_occlusion_augmentation import frozen
from src.models.messenger_natural_task import MessengerTask
from src.models.messenger_occlusion_training import rng_state, restore_rng
from keypoint_comparison1.landmark_detection.geometry_prior_recovery import generator_losses


class CompositionPolicy(nn.Module):
    """Score donor/scale/placement joint proposals, conditioned on face geometry.

No recipient RGB is read, even through alternative proposals. Candidate pools
are fixed before scoring, refreshed per epoch and reproducible from row seeds.
"""
    def __init__(self, hidden=32):
        super().__init__()
        self.visual = nn.Sequential(nn.Conv2d(3, hidden, 3, padding=1), nn.SiLU(),
                                    nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten())
        self.score = nn.Sequential(nn.Linear(hidden*16+15, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.register_buffer('reward_baseline', torch.tensor(0.))

    def forward(self, previews, features, context):
        if previews.ndim != 4 or previews.shape[1:] != (3,16,16) or features.shape != (len(previews),10) or context.shape != (5,):
            raise ValueError('Invalid candidate policy inputs')
        return self.score(torch.cat((self.visual(previews.detach()), features.detach(),
                                    context.detach().expand(len(previews),-1)), -1)).flatten()

    def choose(self, candidates, ellipse, hw, *, action=None):
        device = next(self.parameters()).device
        h, w = hw
        context = ellipse.detach().to(device).float()/torch.tensor([w,h,w,h,math.pi], device=device)
        logits = self(torch.stack([c['preview'] for c in candidates]).to(device),
                      torch.stack([c['features'] for c in candidates]).to(device), context)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample() if action is None else torch.tensor(action,device=device)
        return int(action), distribution.log_prob(action), distribution.entropy()


class FullCropDiscriminator(nn.Module):
    """RGB of the complete crop; spatial 4x4 head retains global arrangement."""
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3,hidden,3,2,1), nn.LeakyReLU(.2),
            nn.Conv2d(hidden,hidden*2,3,2,1), nn.LeakyReLU(.2),
            nn.Conv2d(hidden*2,hidden*4,3,2,1), nn.LeakyReLU(.2),
            nn.AdaptiveAvgPool2d((4,4)), nn.Flatten(), nn.Linear(hidden*4*16,1))

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 32:
            raise ValueError('D requires complete RGB face crops')
        return self.net(images).flatten()


def position_error(coordinates, targets, covered):
    """Per-example Smooth L1 of normalized xy; annotated covered points only.

Presence/state predictions NEVER select or weight this supervision. Coordinates
are crop pixel/(size-1), not the NME normalization used in validation F1.
"""
    truth = torch.stack([t['keypoints'] for t in targets]).to(coordinates).detach()
    valid = torch.stack([t['labeled'] for t in targets]).to(coordinates.device).bool()
    valid = valid & covered.to(coordinates.device) & torch.isfinite(truth).all(-1)
    valid &= (truth >= 0).all(-1) & (truth <= 1).all(-1)
    truth = torch.where(valid[...,None], truth, torch.zeros_like(truth))
    error = F.smooth_l1_loss(coordinates, truth, reduction='none').mean(-1)
    return (error * valid).sum(-1) / valid.sum(-1).clamp_min(1)


class PolicyMessengerTask(MessengerTask):
    """Shared position component for G reward and T supervision.

Existing all-point landmark/reconstruction terms are retained; the explicit
covered-position term has weight +1 for T and -lambda_position for G.
"""
    def output(self, pixels, batch):
        valid = torch.stack([t['labeled'] for t in batch['target']]).to(pixels.device)
        return self.pipeline((pixels-self.mean)/self.std, geometry=batch['geometry'],
            support=batch['support'], augmentation_valid_mask=valid)

    def position(self, pixels, batch):
        return position_error(self.output(pixels,batch)['decoded_coordinates'],batch['target'],batch['covered'])

    def supervised(self, pixels, batch):
        output = self.output(pixels,batch)
        position = position_error(output['decoded_coordinates'],batch['target'],batch['covered']).mean()
        with torch.no_grad(), torch.autocast(pixels.device.type,enabled=False):
            clean = self.pipeline.observer(((batch['clean'].detach()-self.mean)/self.std).float())
        other = generator_losses(output,clean,self.criterion(output,batch['target'])['loss_total'],
                                 reconstruction_weight=self.reconstruction_weight)['total']
        return other + position


def reinforce_loss(log_probs, natural, position, *, natural_weight=1., position_weight=.1, baseline=0.):
    """Minimizing this increases probability of high-position/low-natural-cost actions."""
    if any(not math.isfinite(w) or w < 0 for w in (natural_weight,position_weight)):
        raise ValueError('Objective coefficients must be finite and nonnegative')
    reward = (position_weight*position-natural_weight*natural).detach()
    advantage = reward - torch.as_tensor(baseline,device=reward.device).detach()
    return -(advantage*log_probs).mean(), reward


def select_batch(generator, rows, device):
    images=[]; targets=[]; covered=[]; log_probs=[]; entropy=[]; active=[]; records=[]
    for i,row in enumerate(rows):
        record = dict(row['metadata'], applied=False)
        if row['candidates'] and row['real'] is not None:
            action, log_prob, ent = generator.choose(row['candidates'],
                row['target']['geometry_prior']['predicted_ellipse_crop'],row['clean'].shape[-2:],
                action=row.get('_sampled_action'))
            candidate = row['candidates'][action]
            image,target,mask = compose(row['clean'],candidate['donor'],candidate['mask'],row['target'])
            active.append(i); log_probs.append(log_prob); entropy.append(ent)
            record.update(candidate['metadata']);record.update(applied=True,action=action,
                candidate_count=len(row['candidates']), reason=None,
                log_probability=float(log_prob.detach()))
        else:
            image,target,mask = compose(row['clean'],row['clean'],
                torch.zeros(1,*row['clean'].shape[-2:],dtype=torch.bool),row['target'])
        images.append(image);targets.append(target);covered.append(mask);records.append(record)
    batch = dict(target=targets, covered=torch.stack(covered).to(device),
        clean=torch.stack([r['clean'] for r in rows]).to(device),
        geometry=torch.stack([r['target']['geometry_prior']['geometry'] for r in rows]).to(device),
        support=torch.stack([r['target']['geometry_prior']['support'] for r in rows]).to(device))
    return torch.stack(images).to(device).detach(),batch,active,log_probs,entropy,records


def policy_losses(generator, discriminator, task, rows, *, natural_weight=1., position_weight=.1):
    """All losses use pre-update weights; no G/D/T parameter graph is shared."""
    device = next(generator.parameters()).device
    fake,batch,active,logs,entropy,records = select_batch(generator,rows,device)
    gloss=dloss=None; reward=fake.new_empty(0); natural=position=reward
    if active:
        real=torch.stack([rows[i]['real'] for i in active]).to(device).detach()
        if real.shape != fake[active].shape:
            raise ValueError('Real/fake crop sizes differ')
        dloss=(F.softplus(-discriminator(real))+F.softplus(discriminator(fake[active].detach()))).mean()
        with frozen(discriminator), frozen(task), torch.no_grad():
            natural=F.softplus(-discriminator(fake[active]))
            position=task.position(fake,batch)[active]
        gloss,reward=reinforce_loss(torch.stack(logs),natural,position,
            natural_weight=natural_weight,position_weight=position_weight,baseline=generator.reward_baseline.clone())
    tloss=task.supervised(fake.detach(),batch)
    return dict(g=gloss,d=dloss,t=tloss,reward=reward,natural=natural,position=position,
                entropy=torch.stack(entropy).mean() if entropy else fake.new_zeros(()),records=records,active=len(active))


class PolicyUpdate:
    """Effective-batch accumulation; precommit OOM replays sampling and buffers."""
    def __init__(self, generator, discriminator, task, optimizers, *, natural_weight=1., position_weight=.1):
        self.generator=generator;self.discriminator=discriminator;self.task=task
        self.optimizers=optimizers;self.weights=dict(natural_weight=natural_weight,position_weight=position_weight)
        self.modules={'g':generator,'d':discriminator,'t':task};self.step=0;self.needs_restore=False
        if set(optimizers)!=set(self.modules):raise ValueError('Require independent G/D/T optimizers')
        owned=set()
        for name,module in self.modules.items():
            ps=[p for group in optimizers[name].param_groups for p in group['params']]
            ids={id(p) for p in ps}
            if len(ids)!=len(ps) or ids & owned or ids!={id(p) for p in module.parameters() if p.requires_grad}:
                raise ValueError('Incorrect/shared optimizer ownership')
            owned |= ids
        if any(p.requires_grad for m in (task.pipeline.bank,task.pipeline.observer) for p in m.parameters()):
            raise ValueError('O and observer must be frozen')

    def update(self, rows, *, microbatch):
        if self.needs_restore:raise RuntimeError('Restore full checkpoint before continuing')
        if not rows or microbatch<1:raise ValueError('Empty/invalid effective batch')
        # Keep SUCCESSFUL synthesis exactly balanced, including when proposals
        # or real matches fail. Excess rows train T on their natural image.
        rows=[dict(r,metadata=dict(r['metadata'])) for r in rows]
        groups=[[i for i,r in enumerate(rows) if r['metadata'].get('bin')==b
                 and r['candidates'] and r['real'] is not None] for b in range(4)]
        quota=min(map(len,groups));accepted={i for group in groups for i in group[:quota]}
        for i,row in enumerate(rows):
            if row['candidates'] and row['real'] is not None and i not in accepted:
                row['candidates']=[];row['metadata']['reason']='bin_balance_deferred'
        # Sample all actions before T consumes dropout RNG. OOM retries and
        # microbatch splits reuse the same discrete actions, recomputing log p.
        with torch.no_grad():
            for row in rows:
                if row['candidates'] and row['real'] is not None:
                    row['_sampled_action']=self.generator.choose(row['candidates'],
                        row['target']['geometry_prior']['predicted_ellipse_crop'],row['clean'].shape[-2:],
                        action=row.get('_sampled_action'))[0]
        state=rng_state();buffers={k:{n:b.detach().clone() for n,b in m.named_buffers()} for k,m in self.modules.items()}
        active_total=sum(bool(r['candidates']) and r['real'] is not None for r in rows)
        retries=[]
        while True:
            for opt in self.optimizers.values():opt.zero_grad(set_to_none=True)
            for module in self.modules.values():module.train()
            committing=False;records=[];totals={};reward_sum=0.
            try:
                for start in range(0,len(rows),microbatch):
                    sub=rows[start:start+microbatch]
                    losses=policy_losses(self.generator,self.discriminator,self.task,sub,**self.weights)
                    for name in ('d','g','t'):
                        value=losses[name]
                        if value is None:continue
                        if not torch.isfinite(value):raise FloatingPointError('Nonfinite '+name)
                        weight=len(sub)/len(rows) if name=='t' else losses['active']/max(1,active_total)
                        (value*weight).backward()
                        totals[name]=totals.get(name,0.)+float(value.detach())*weight
                    records.extend(losses['records']);reward_sum+=float(losses['reward'].sum())
                    for key in ('natural','position'):
                        totals[key]=totals.get(key,0.)+float(losses[key].sum())/max(1,active_total)
                names=('g','d','t') if active_total else ('t',)
                norms={name:float(torch.nn.utils.clip_grad_norm_(
                    [p for group in self.optimizers[name].param_groups for p in group['params']],1.,error_if_nonfinite=True)) for name in names}
                if any(p.grad is not None for m in (self.task.pipeline.bank,self.task.pipeline.observer) for p in m.parameters()):
                    raise AssertionError('Gradient escaped into observer/O')
                committing=True
                for name in names:self.optimizers[name].step()
                if active_total:
                    self.generator.reward_baseline.mul_(.9).add_(.1*reward_sum/active_total)
                self.step+=1
                for opt in self.optimizers.values():opt.zero_grad(set_to_none=True)
                return dict(status='updated',step=self.step,microbatch=microbatch,retries=retries,
                    losses=totals,gradient_norms=norms,g_updated=bool(active_total),d_updated=bool(active_total),records=records)
            except torch.OutOfMemoryError:
                if committing:self.needs_restore=True;raise
                for opt in self.optimizers.values():opt.zero_grad(set_to_none=True)
                losses=value=None
                with torch.no_grad():
                    for key,module in self.modules.items():
                        for name,buffer in module.named_buffers():buffer.copy_(buffers[key][name])
                restore_rng(state)
                if next(self.generator.parameters()).is_cuda:torch.cuda.empty_cache()
                if microbatch==1:raise
                retries.append(microbatch);microbatch=max(1,microbatch//2)
            except BaseException:
                self.needs_restore=True;raise
