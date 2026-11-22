"""Two authorized E16/K8192 conditions; synthetic-mask teacher only."""
import torch
from src.models.messenger_codebook_reconstruction import ReconstructionRecovery
from src.models.messenger_code_consistency import ScaleRecovery, ConsistencyRecovery

ARMS=('E16K8192_s124','E16K8192_s1248_js001')
def arm_spec(arm):
    if arm not in ARMS: raise ValueError(arm)
    return dict(dimension=16,count=8192,update='ema',normalized=True,spatial=True,
                scales=[1,2,4] if arm==ARMS[0] else [1,2,4,8],
                js_weight=0.0 if arm==ARMS[0] else 0.01,temperature=0.1)

class FullERecovery(ReconstructionRecovery):
    selection_statistics=ConsistencyRecovery.selection_statistics
    def __init__(self,arm,basis,mean,centers):
        super().__init__('E_d16_k8192',basis,mean,centers)
        spec=arm_spec(arm)
        self.messenger=ScaleRecovery(spec['scales'])
        self.js_weight=spec['js_weight'];self.temperature=spec['temperature']
    def forward(self,maps,covered=None):
        fused,candidates,_=self.messenger(maps)
        latent,details=self.codebook(self.encoder(fused),covered)
        restored=self.decoder(latent)+self.decoder_residual(latent)+self.spatial(latent)
        if covered is not None:
            if self.js_weight>0 and torch.is_grad_enabled():
                details.update(self.selection_statistics(candidates,covered))
            else:
                with torch.no_grad():details.update(self.selection_statistics(candidates,covered))
        return restored,details

def make_recovery(arm,basis,mean,centers):
    return FullERecovery(arm,basis,mean,centers)
