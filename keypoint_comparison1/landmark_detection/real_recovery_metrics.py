"""Token-weighted paired recovery diagnostics, separated by real occluder family.

Feed the SAME fixed teacher's clean features and student's raw/candidate/final
features. Positive cosine gain and reduced normalized MSE indicate feature
recovery on these pairs; they do not establish downstream or validation benefit.
"""
import torch
import torch.nn.functional as F
from .messenger_recovery import patch_occlusion_ratios


class RealRecoveryMeter:
    def __init__(self):
        self.totals={}

    @torch.no_grad()
    def update(self,diagnostics,clean_map,pixel_masks,targets,*,patch_size=16,channel_scale=None):
        clean=clean_map.detach().float().flatten(2).transpose(1,2)
        features={name:diagnostics[key].detach().float() for name,key in
                  [('raw','original_tokens'),('candidate','candidate_tokens'),('final','recovered_tokens')]}
        if any(x.shape!=clean.shape for x in features.values()):raise ValueError('Feature shapes differ')
        if not all(torch.isfinite(x).all() for x in [clean,*features.values()]):raise ValueError('Nonfinite features')
        ratios=patch_occlusion_ratios(pixel_masks,patch_size=patch_size).to(clean.device)
        scale=torch.ones(clean.shape[-1],device=clean.device) if channel_scale is None else channel_scale.detach().to(clean).clamp_min(.05)
        cos={k:F.cosine_similarity(v,clean,dim=-1) for k,v in features.items()}
        mse={k:((v-clean)/scale).square().mean(-1) for k,v in features.items()}
        for i,target in enumerate(targets):
            meta=target.get('paired_occlusion',{})
            if not meta.get('applied'):continue
            family=meta['requested_family']
            if family not in {'face','non_face'}:raise ValueError('Unknown real occlusion family')
            for region,mask in [('masked',ratios[i]>0),('unmasked',ratios[i]==0)]:
                n=int(mask.sum())
                if not n:continue
                for group in (family,'all'):
                    row=self.totals.setdefault(f'{group}/{region}',{'tokens':0,'improved_tokens':0})
                    row['tokens']+=n
                    row['improved_tokens']+=int((cos['final'][i,mask]>cos['raw'][i,mask]).sum())
                    for name in features:
                        for metric,values in [('cosine',cos),('normalized_mse',mse)]:
                            key=f'{name}_{metric}'
                            row[key]=row.get(key,0.)+float(values[name][i,mask].sum())

    def report(self):
        out={}
        for family in ('face','non_face','all'):
            for region in ('masked','unmasked'):
                key=f'{family}/{region}';row=self.totals.get(key)
                if row is None:
                    out[key]={'tokens':0,'metrics':None};continue
                n=row['tokens'];metrics={k:v/n for k,v in row.items() if k not in {'tokens','improved_tokens'}}
                metrics['final_minus_raw_cosine']=metrics['final_cosine']-metrics['raw_cosine']
                metrics['final_minus_raw_normalized_mse']=metrics['final_normalized_mse']-metrics['raw_normalized_mse']
                metrics['improved_token_fraction']=row['improved_tokens']/n
                out[key]={'tokens':n,'metrics':metrics}
        return out
