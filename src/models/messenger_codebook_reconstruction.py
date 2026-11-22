"""Predict a complete 384-D feature before compression; no input-feature skip."""
import torch
from torch import nn
from torch.nn import functional as F
from src.models.messenger_multiaxis import MultiAxisRecovery
from src.models.messenger_codebook_unit import UnitVectorCodebook
from src.models.messenger_codebook_factorial import SpatialDecoder, arm_spec as vq_spec

DIMENSIONS=(16,32)
COUNTS=(4096,8192)
ARMS=('continuous32','regression384','continuous16')+tuple(
    f'{m}_d{d}_k{k}' for d in DIMENSIONS for k in COUNTS for m in 'BDE')

def arm_spec(arm):
    if arm in ('continuous16','continuous32','regression384'):
        return dict(dimension=int(arm.replace('continuous','').replace('regression','')),
                    count=0,normalized=False,update='none',spatial=False,mode=arm)
    return vq_spec(arm)

class CoveredCodebook(UnitVectorCodebook):
    def forward(self,latent,covered=None):
        # Evaluate selection without accumulating all positions. EMA only sees masked tokens.
        training=self.training
        self.training=False
        try:
            quantized,details=super().forward(latent)
        finally:
            self.training=training
        if training and self.update=='ema':
            if covered is None or covered.shape!=(latent.shape[0],*latent.shape[2:]):
                raise ValueError('Training EMA requires the aligned teacher mask')
            vectors=F.normalize(latent.detach(),dim=1,eps=1e-6).permute(0,2,3,1)
            self.accumulate(vectors[covered.bool()],details['indices'][covered.bool()])
        return quantized,details

class ReconstructionRecovery(nn.Module):
    def __init__(self,arm,basis,mean,centers=None):
        super().__init__()
        spec=arm_spec(arm);d=spec['dimension']
        self.messenger=MultiAxisRecovery('scale')
        self.codebook=None
        self.spatial=None
        self.dimension=d
        if d==384:
            self.encoder=nn.Identity();self.decoder=nn.Identity()
            return
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.encoder=nn.Conv2d(384,d,1)
            self.decoder=nn.Conv2d(d,384,1)
            self.decoder_residual=nn.Sequential(nn.Conv2d(d,96,1),nn.GELU(),nn.Conv2d(96,384,1))
            nn.init.zeros_(self.decoder_residual[-1].weight)
            nn.init.zeros_(self.decoder_residual[-1].bias)
        with torch.no_grad():
            self.encoder.weight.copy_(basis[:,:d].T[:,:,None,None])
            self.encoder.bias.copy_(-basis[:,:d].T@mean)
            self.decoder.weight.copy_(basis[:,:d][:,:,None,None])
            self.decoder.bias.copy_(mean)
        if spec['count']:
            if centers is None or centers.shape!=(spec['count'],d):
                raise ValueError('Incorrect codebook shape')
            self.codebook=CoveredCodebook(centers,spec['update'])
        if spec['spatial']:
            self.spatial=SpatialDecoder(d)

    def predict_features(self,maps):
        return self.messenger(maps)[0]

    def encode(self,maps):
        return self.encoder(self.predict_features(maps))

    def forward(self,maps,covered=None):
        latent=self.encode(maps)
        details={}
        if self.codebook is not None:
            latent,details=self.codebook(latent,covered)
        restored=self.decoder(latent)
        if self.dimension!=384:
            restored=restored+self.decoder_residual(latent)
        if self.spatial is not None:
            restored=restored+self.spatial(latent)
        return restored,details

def make_recovery(arm,basis,mean,centers=None):
    return ReconstructionRecovery(arm,basis,mean,centers)
