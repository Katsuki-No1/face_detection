"""Epoch/validation/checkpoint integration for the opt-in geometry-prior runtime.

This is a library, not authorization to launch production without authenticated
gates and isolated preflight. Validation is injected and must return the fixed
conditional semantic metric. Detailed test evaluation belongs to another job.
"""
import copy
import math
from pathlib import Path
import time

import torch

from .training import update_early_stopping_patience

METRIC = 'conditional_semantic_macro_f1_nme_0.10'


class PriorEpochDriver:
    def __init__(self, runtime, *, schedulers, loader_generator, effective_batch=64,
                 microbatch=4, freeze_backbone_epochs=5, mixing_warmup_epochs=2,
                 minimum_epoch=10, patience=5, min_delta=1e-4, maximum_epochs=40,
                 maximum_consecutive_overflows=8):
        if min(effective_batch,microbatch,patience,maximum_epochs,minimum_epoch,maximum_consecutive_overflows)<1:
            raise ValueError('Invalid epoch/batch contract')
        if min(freeze_backbone_epochs,mixing_warmup_epochs,minimum_epoch)<0 or not math.isfinite(min_delta) or min_delta<0:
            raise ValueError('Invalid epoch policy')
        if set(schedulers)-{'g','d'} or 'g' not in schedulers:
            raise ValueError('Generator scheduler required; only G/D schedulers supported')
        if ('d' in schedulers) != (runtime.discriminator is not None):
            raise ValueError('D scheduler contract differs from runtime')
        self.runtime=runtime;self.schedulers=schedulers;self.loader_generator=loader_generator
        self.contract=dict(effective_batch=effective_batch,initial_microbatch=microbatch,
            freeze_backbone_epochs=freeze_backbone_epochs,mixing_warmup_epochs=mixing_warmup_epochs,
            minimum_epoch=minimum_epoch,patience=patience,min_delta=min_delta,maximum_epochs=maximum_epochs,
            maximum_consecutive_overflows=maximum_consecutive_overflows,metric=METRIC,
            scheduler_step='G after committed G update; D only after committed D update')
        self.microbatch=microbatch;self.epoch=0;self.best=-math.inf;self.best_epoch=0
        self.no_improvement=0;self.overflow_count=0;self.last_row=None

    def run_epoch(self, loader, validate, *, logger=None):
        if self.runtime.requires_checkpoint_restore:
            raise RuntimeError('Restore a complete checkpoint before retrying an epoch')
        try:
            return self._run_epoch(loader,validate,logger=logger)
        except Exception:
            # Validation/logging may fail after real optimizer updates. Never
            # replay the epoch over those partially advanced weights.
            self.runtime.requires_checkpoint_restore=True
            raise

    def _run_epoch(self, loader, validate, *, logger=None):
        if self.epoch>=self.contract['maximum_epochs'] or self.should_stop:
            raise RuntimeError('Training already reached its stopping boundary')
        if loader.batch_size!=self.contract['effective_batch'] or loader.drop_last:
            raise ValueError('Loader must preserve complete effective windows, including final partial window')
        if loader.generator is not self.loader_generator or loader.persistent_workers:
            raise ValueError('Resume requires the checkpointed loader generator and nonpersistent workers')
        epoch=self.epoch+1
        dataset=loader.dataset
        while isinstance(dataset,torch.utils.data.Subset):dataset=dataset.dataset
        if hasattr(dataset,'set_epoch'):dataset.set_epoch(epoch)
        frozen=epoch<=self.contract['freeze_backbone_epochs']
        for p in self.runtime.model.base.backbone.parameters():p.requires_grad_(not frozen)
        self.runtime.model.base.messenger_recovery.mixing_enabled=epoch>self.contract['mixing_warmup_epochs']
        device=self.runtime.device
        if device.type=='cuda':torch.cuda.synchronize(device)
        start=time.perf_counter();totals={};committed_samples=0;seen=0;overflows=0;consecutive=0
        for images,targets in loader:
            n=len(images);seen+=n
            clean=torch.stack([t['clean_image'] for t in targets])
            geometry=torch.stack([t['geometry_prior']['geometry'] for t in targets])
            support=torch.stack([t['geometry_prior']['support'] for t in targets])
            result=self.runtime.update_window(images,clean,geometry,support,targets,microbatch=self.microbatch)
            # Do not let a final short window change the persistent batch plan.
            if result['attempts']:self.microbatch=min(self.microbatch,result['microbatch'])
            if result['status']=='overflow_skipped_both':
                overflows+=1;consecutive+=1
                if consecutive>self.contract['maximum_consecutive_overflows']:
                    raise FloatingPointError('Too many consecutive AMP overflows; restore last epoch')
                continue
            consecutive=0;committed_samples+=n
            self.schedulers['g'].step()
            if result['discriminator_updated']:self.schedulers['d'].step()
            for key,value in result['losses'].items():totals[key]=totals.get(key,0.)+value*n
        if seen!=len(loader.dataset) or not committed_samples:
            raise RuntimeError('Incomplete epoch or no optimizer updates')
        if device.type=='cuda':torch.cuda.synchronize(device)
        training_seconds=time.perf_counter()-start
        validation_started=time.perf_counter()
        # Validation must not perturb the next training epoch RNG.
        from .geometry_prior_training import rng_state,restore_rng
        saved_rng=rng_state()
        try:metrics=validate(self.runtime.ema.module)
        finally:restore_rng(saved_rng)
        if device.type=='cuda':torch.cuda.synchronize(device)
        value=float(metrics[METRIC])
        if not math.isfinite(value) or not 0<=value<=1 or metrics.get('eligible_face_crops',0)<=0:
            raise ValueError('Missing/non-finite conditional validation metric')
        improved=value-self.best>self.contract['min_delta']
        if improved:self.best=value;self.best_epoch=epoch
        self.no_improvement=update_early_stopping_patience(epoch=epoch,improved=improved,
            minimum_epoch=self.contract['minimum_epoch'],no_improvement_validations=self.no_improvement)
        self.epoch=epoch;self.overflow_count+=overflows
        row=dict(epoch=epoch,step=self.runtime.step,backbone_frozen=frozen,
            train={k:v/committed_samples for k,v in totals.items()},train_seen=seen,
            committed_samples=committed_samples,amp_overflows=overflows,microbatch=self.microbatch,
            effective_batch=self.contract['effective_batch'],validation=metrics,
            train_seconds=training_seconds,validation_seconds=time.perf_counter()-validation_started,
            best_epoch=self.best_epoch,best_value=self.best,improved=improved,
            no_improvement_validations=self.no_improvement,should_stop=self.should_stop,
            learning_rates={g.get('name',str(i)):g['lr'] for i,g in enumerate(self.runtime.optimizer.param_groups)})
        self.last_row=row
        if logger is not None:logger(row)
        return row

    @property
    def should_stop(self):
        return self.epoch>=self.contract['minimum_epoch'] and self.no_improvement>=self.contract['patience']

    def checkpoint(self):
        if self.epoch<1 or self.last_row is None:raise RuntimeError('No completed validated epoch to save')
        state=self.runtime.checkpoint(epoch=self.epoch,schedulers=self.schedulers,loader_generator=self.loader_generator)
        state['epoch_driver']=dict(contract=copy.deepcopy(self.contract),microbatch=self.microbatch,
            best=self.best,best_epoch=self.best_epoch,no_improvement=self.no_improvement,
            overflow_count=self.overflow_count,last_row=copy.deepcopy(self.last_row))
        return state

    def restore(self,state):
        policy=state.get('epoch_driver',{})
        if policy.get('contract')!=self.contract:raise ValueError('Epoch contract changed')
        epoch=self.runtime.restore(state,schedulers=self.schedulers,loader_generator=self.loader_generator)
        if epoch!=policy['last_row']['epoch']:raise ValueError('Inconsistent epoch checkpoint')
        self.epoch=epoch;self.microbatch=policy['microbatch'];self.best=policy['best']
        self.best_epoch=policy['best_epoch'];self.no_improvement=policy['no_improvement']
        self.overflow_count=policy['overflow_count'];self.last_row=copy.deepcopy(policy['last_row'])


def save_epoch(driver, directory):
    """Atomic last every epoch, best on improvement, one rotating >=5-epoch snapshot."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    state=driver.checkpoint()
    def save(name):
        temp=directory/(name+'.tmp');torch.save(state,temp);temp.replace(directory/name)
    # Save best first so a crash can never leave last claiming a missing best.
    if driver.last_row['improved']:save('best.pt')
    save('last.pt')
    if driver.epoch%5==0:
        name=f'epoch_{driver.epoch:03d}.pt';save(name)
        for path in directory.glob('epoch_[0-9][0-9][0-9].pt'):
            if path.name!=name:path.unlink()


def finalize_checkpoints(driver,directory):
    if not (driver.should_stop or driver.epoch==driver.contract['maximum_epochs']):
        raise RuntimeError('Cannot finalize an incomplete training run')
    directory=Path(directory)
    if not (directory/'last.pt').is_file() or not (directory/'best.pt').is_file():
        raise RuntimeError('Final last/best checkpoint missing')
    for path in directory.glob('epoch_[0-9][0-9][0-9].pt'):path.unlink()


def wandb_epoch_logger(run):
    """Log scalar epoch counters only; W&B run setup remains authenticated caller work."""
    if run is None:raise ValueError('Production W&B run is required')
    def log(row):
        values={k:v for k,v in row.items() if isinstance(v,(int,float,bool))}
        for namespace in ['train','validation','learning_rates']:
            values.update({f'{namespace}/{k}':v for k,v in row[namespace].items() if isinstance(v,(int,float,bool))})
        run.log(values,step=row['step'])
    return log
