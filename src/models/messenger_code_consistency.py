"""Shared-codebook scale consistency; main hard lookup stays unchanged."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.models.messenger_multiaxis import Branch
from src.models.messenger_codebook_reconstruction import ReconstructionRecovery

DIMENSIONS=(16,)
COUNTS=(4096,)
ARMS=('s124_js0','s124_js001','s1248_js0','s1248_js001')
TRAIN_ARMS=ARMS[1:]
CONTROL_ARM=ARMS[0]
TEMPERATURE=0.1

def arm_spec(arm):
    if arm not in ARMS: raise ValueError(arm)
    return dict(dimension=16,count=4096,normalized=True,update='gradient',spatial=False,
                scales=[1,2,4,8] if arm.startswith('s1248_') else [1,2,4],
                js_weight=0.01 if arm.endswith('js001') else 0.0,temperature=TEMPERATURE)

class ScaleRecovery(nn.Module):
    def __init__(self,scales):
        super().__init__();self.specs=[(2,s) for s in scales];branches=[]
        for scale in scales:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42+200+scale)
                branches.append(Branch(384,96,scale))
        self.branches=nn.ModuleList(branches)
    def forward(self,maps):
        values,scores=zip(*(branch(maps[:,level]) for branch,(level,_) in zip(self.branches,self.specs)))
        candidates=torch.stack(values,1);weights=torch.stack(scores,1).softmax(1)
        return (candidates*weights).sum(1),candidates,weights

class ConsistencyRecovery(ReconstructionRecovery):
    def __init__(self,arm,basis,mean,centers):
        super().__init__('B_d16_k4096',basis,mean,centers)
        spec=arm_spec(arm);self.messenger=ScaleRecovery(spec['scales'])
        self.js_weight=spec['js_weight'];self.temperature=spec['temperature']

    def selection_statistics(self,candidates,covered):
        """Per-position soft lookup only: never call the VQ update path here."""
        b,s,c,h,w=candidates.shape
        z=self.encoder(candidates.reshape(b*s,c,h,w)).reshape(b,s,self.dimension,h,w)
        z=F.normalize(z,dim=2,eps=1e-6).permute(0,3,4,1,2).reshape(-1,s,self.dimension)
        if covered is None: raise ValueError('Consistency needs aligned teacher coverage')
        z=z[covered.reshape(-1).bool()]
        codes=F.normalize(self.codebook.embedding.detach(),dim=1,eps=1e-6)
        if len(z)==0:
            zero=candidates.sum()*0
            return dict(js=zero,branch_entropy=zero,fused_consensus_entropy=zero,branch_code_agreement=zero,covered_tokens=0)
        js=[];ent=[];cons_ent=[];agree=[]
        for chunk in z.split(128):
            logp=(chunk@codes.T/self.temperature).log_softmax(-1)
            p=logp.exp();logm=torch.logsumexp(logp,dim=1)-math.log(s)
            js.append((p*(logp-logm[:,None])).sum(-1).mean(1))
            with torch.no_grad():
                ent.append(-(p*logp).sum(-1).mean(1))
                cons_ent.append(-(logm.exp()*logm).sum(-1))
                ids=logp.argmax(-1)
                agree.append(torch.stack([(ids[:,i]==ids[:,j]).float() for i in range(s) for j in range(i+1,s)],1).mean(1))
        return dict(js=torch.cat(js).mean(),branch_entropy=torch.cat(ent).mean(),
                    fused_consensus_entropy=torch.cat(cons_ent).mean(),branch_code_agreement=torch.cat(agree).mean(),covered_tokens=len(z))

    def forward(self,maps,covered=None):
        fused,candidates,weights=self.messenger(maps)
        z=self.encoder(fused);latent,details=self.codebook(z,covered)
        restored=self.decoder(latent)+self.decoder_residual(latent)
        # Soft distributions are training/validation diagnostics; inference stays hard.
        if covered is not None:
            if self.js_weight>0 and torch.is_grad_enabled():
                statistics=self.selection_statistics(candidates,covered)
            else:
                with torch.no_grad():statistics=self.selection_statistics(candidates,covered)
            details.update(statistics)
        return restored,details

def make_recovery(arm,basis,mean,centers=None):
    return ConsistencyRecovery(arm,basis,mean,centers)
