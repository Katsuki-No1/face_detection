"""Conditional feature-space v-diffusion with independent scale attention branches."""
import math
import torch
from torch import nn
from keypoint_comparison1.landmark_detection.heads import sine_position_encoding_2d
from src.models.messenger_multiaxis import MultiAxisRecovery


def coefficients(t):
    angle=t*math.pi/2
    return angle.cos()[:,None,None,None],angle.sin()[:,None,None,None]


def noise_for(ids,shape,device,seed=42):
    # Identity-bound sampling: invariant to batching and condition ordering.
    return torch.stack([torch.randn(shape,generator=torch.Generator().manual_seed(seed+int(i)*100003)) for i in ids]).to(device)


class FeatureDiffusion(MultiAxisRecovery):
    def __init__(self,mean,std):
        super().__init__('scale')
        self.register_buffer('mean',mean.reshape(1,384,1,1))
        self.register_buffer('std',std.reshape(1,384,1,1))
        self.noisy=nn.Linear(384,96)
        self.time=nn.Sequential(nn.Linear(32,96),nn.SiLU(),nn.Linear(96,96))

    def denoise(self,maps,xt,t):
        b,c,h,w=xt.shape
        freq=torch.exp(torch.arange(16,device=xt.device)*(-math.log(10000)/15))
        angle=t[:,None]*1000*freq
        time=self.time(torch.cat((angle.sin(),angle.cos()),1))[:,None]
        noisy=self.noisy(xt.flatten(2).transpose(1,2))
        values=[];scores=[]
        for branch,(level,scale) in zip(self.branches,self.specs):
            memory=branch.patch(maps[:,level]);mh,mw=memory.shape[-2:]
            memory=memory.flatten(2).transpose(1,2)
            kp=sine_position_encoding_2d(mh,mw,96,device=xt.device,dtype=xt.dtype)
            qp=sine_position_encoding_2d(h,w,96,device=xt.device,dtype=xt.dtype)
            q=branch.query.expand(b,h*w,-1)+qp+noisy+time
            for layer in branch.layers:
                k=layer['knorm'](memory)
                q=q+layer['attn'](layer['qnorm'](q),k+kp,k,need_weights=False)[0]
                q=q+layer['ffn'](layer['norm'](q))
            values.append(branch.decode(q).transpose(1,2).reshape(b,c,h,w))
            scores.append(branch.score(q).transpose(1,2).reshape(b,1,h,w))
        values=torch.stack(values,1);weights=torch.stack(scores,1).softmax(1)
        return (values*weights).sum(1),values,weights

    def training_pair(self,maps,clean,t,noise):
        z=(clean-self.mean)/self.std;a,s=coefficients(t)
        xt=a*z+s*noise;target=a*noise-s*z
        outputs=self.denoise(maps,xt,t)
        restored=(a*xt-s*outputs[0])*self.std+self.mean
        return outputs,target,restored

    def sample(self,maps,ids,seed=42,steps=16):
        xt=noise_for(ids,maps[:,2].shape[1:],maps.device,seed)
        for step in range(steps,0,-1):
            t=torch.full((len(maps),),step/steps,device=maps.device)
            a,s=coefficients(t);v,_,_=self.denoise(maps,xt,t)
            clean=a*xt-s*v;noise=s*xt+a*v
            na,ns=coefficients(t-1/steps);xt=na*clean+ns*noise
        return xt*self.std+self.mean
