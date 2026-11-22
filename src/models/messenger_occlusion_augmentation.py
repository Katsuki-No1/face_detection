"""Train-only bounded RGB generator and point-region naturalness discriminator.

Placement/masks are fixed per paired sample. No learned geometry or D_feature.
G never sees recipient pixels under the paste; no soft alpha compositing.
"""
from contextlib import contextmanager
import math

import torch
from torch import nn
import torch.nn.functional as F


@contextmanager
def frozen(module):
    parameters = [(p, p.requires_grad) for p in module.parameters()]
    modes = [(m, m.training) for m in module.modules()]
    try:
        module.requires_grad_(False).eval()
        yield
    finally:
        for p, flag in parameters: p.requires_grad_(flag)
        for m, flag in modes: m.training = flag


def validate_pixels(recipient, donor, mask):
    if recipient.ndim != 4 or recipient.shape[1] != 3 or donor.shape != recipient.shape:
        raise ValueError('Expected paired Bx3xHxW RGB')
    if mask.shape != (len(recipient), 1, *recipient.shape[-2:]) or mask.dtype != torch.bool:
        raise ValueError('Binary Bx1xHxW paste mask required')
    if not torch.isfinite(recipient).all() or not torch.isfinite(donor).all():
        raise ValueError('Nonfinite RGB')
    if (recipient < 0).any() or (recipient > 1).any() or (donor < 0).any() or (donor > 1).any():
        raise ValueError('RGB must be in [0,1]')


class BoundedOccluderRGB(nn.Module):
    """Low-frequency RGB residual, bounded in amplitude, initialized to identity.

    A 4x4 residual field prevents arbitrary per-pixel noise. Its input contains
    only masked donor, visible recipient and binary mask. Bilinear interpolation
    smooths the correction field, not the pasted alpha or donor image edges.
    """
    def __init__(self, max_rgb_delta=.12, field_size=4, hidden=32):
        super().__init__()
        if not 0 < max_rgb_delta <= .25 or not 1 <= field_size <= 8:
            raise ValueError('Invalid bounded RGB correction')
        self.max_rgb_delta = max_rgb_delta; self.field_size = field_size
        self.net = nn.Sequential(nn.Conv2d(7,hidden,3,padding=1),nn.SiLU(),
            nn.Conv2d(hidden,hidden,3,padding=1),nn.SiLU(),nn.Conv2d(hidden,3,1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def get_extra_state(self):
        return dict(max_rgb_delta=self.max_rgb_delta, field_size=self.field_size)

    def set_extra_state(self, state):
        if state != self.get_extra_state(): raise ValueError('RGB constraints changed')

    def forward(self, recipient, donor, mask):
        validate_pixels(recipient, donor, mask)
        # Detach all sources; train G parameters only during G updates.
        recipient = recipient.detach(); donor = donor.detach(); mask = mask.detach()
        visible = torch.where(mask, torch.zeros_like(recipient), recipient)
        material = torch.where(mask, donor, torch.zeros_like(donor))
        context = F.adaptive_avg_pool2d(torch.cat((visible,material,mask.float()),1), (16,16))
        low = F.adaptive_avg_pool2d(self.net(context), self.field_size).tanh()*self.max_rgb_delta
        delta = F.interpolate(low, recipient.shape[-2:], mode='bilinear', align_corners=False)
        corrected = (donor+delta).clamp(0,1)
        output = torch.where(mask, corrected, recipient)
        return dict(image=output, delta=delta, donor=donor, mask=mask)


def preservation_loss(result):
    delta = result['delta']; mask = result['mask'].float()
    amplitude = (delta.square()*mask).sum()/(3*mask.sum()).clamp_min(1)
    smooth = delta.diff(dim=-1).square().mean()+delta.diff(dim=-2).square().mean()
    return amplitude+smooth


def point_patches(images, xy, *, patch_size=32):
    """Same bilinear point-centered crop for real and fake; reject crop-edge patches.

    xy has shape Bx2 in crop pixel coordinates. Return patches and valid indices.
    No fake mask or hidden clean target is supplied to D.
    """
    if images.ndim != 4 or xy.shape != (len(images),2) or patch_size < 2:
        raise ValueError('Invalid point-region input')
    h,w = images.shape[-2:]; half=(patch_size-1)/2
    valid = torch.isfinite(xy).all(1) & (xy[:,0]>=half) & (xy[:,1]>=half)
    valid &= (xy[:,0]<=w-1-half) & (xy[:,1]<=h-1-half)
    index = valid.nonzero().flatten()
    if not len(index): return images.new_empty((0,3,patch_size,patch_size)), index
    offset = torch.arange(patch_size,device=images.device,dtype=images.dtype)-half
    yy,xx = torch.meshgrid(offset,offset,indexing='ij')
    pixels = torch.stack((xx,yy),-1)[None]+xy[index,None,None,:].detach()
    grid = pixels/pixels.new_tensor([w-1,h-1])*2-1
    return F.grid_sample(images[index],grid,mode='bilinear',padding_mode='zeros',align_corners=True), index


class NaturalOcclusionDiscriminator(nn.Module):
    """Classify matched occluded-point RGB patches; semantic slot is symmetric."""
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(8,hidden,3,stride=2,padding=1),nn.LeakyReLU(.2),
            nn.Conv2d(hidden,hidden*2,3,stride=2,padding=1),nn.LeakyReLU(.2),
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(hidden*2,1))

    def forward(self, patches, slots):
        if patches.ndim != 4 or slots.shape != (len(patches),) or ((slots<0)|(slots>=5)).any():
            raise ValueError('Invalid D patch/slot condition')
        onehot = F.one_hot(slots.long(),5).to(patches)[:,:,None,None].expand(-1,-1,*patches.shape[-2:])
        return self.net(torch.cat((patches,onehot),1)).flatten()


def matched_regions(fake, real, fake_xy, real_xy, slots, valid, *, patch_size=32):
    # Joint rejection prevents fake-only edge padding from becoming a shortcut.
    _, fi = point_patches(fake, fake_xy, patch_size=patch_size)
    _, ri = point_patches(real, real_xy, patch_size=patch_size)
    keep = torch.zeros(len(fake),dtype=torch.bool,device=fake.device)
    keep[fi] = True
    other = torch.zeros_like(keep); other[ri] = True
    ids = (keep & other & valid.bool()).nonzero().flatten()
    fp,_ = point_patches(fake[ids],fake_xy[ids],patch_size=patch_size)
    rp,_ = point_patches(real[ids],real_xy[ids],patch_size=patch_size)
    return fp,rp,slots[ids],ids


def separated_losses(generator, discriminator, task, batch, *, realism_weight=1.,
                     preserve_weight=10., hardness_weight=.1, patch_size=32):
    """Build disjoint G/D/T graphs at the same preupdate weights.

    task.loss(image, batch, hardness=False) returns a scalar. For hardness=True
    it must include covered-point coordinate and presence losses. T parameters
    are frozen, but its input derivative is retained. task owns normalization.
    """
    if any(not math.isfinite(v) or v < 0 for v in (realism_weight,preserve_weight,hardness_weight)):
        raise ValueError('Invalid objective weight')
    result = generator(batch['clean'],batch['donor'],batch['mask'])
    fake = result['image']
    fp,rp,slots,ids = matched_regions(fake,batch['real'],batch['fake_xy'],batch['real_xy'],
        batch['slots'],batch['real_valid'],patch_size=patch_size)
    dloss = None; realism = fake.sum()*0
    if len(ids):
        dloss = (F.softplus(-discriminator(rp.detach(),slots))+F.softplus(discriminator(fp.detach(),slots))).mean()
    with frozen(discriminator), frozen(task):
        if len(ids): realism = F.softplus(-discriminator(fp,slots)).mean()
        hard = task.loss(fake,batch,hardness=True) if hardness_weight else fake.sum()*0
    gloss = realism_weight*realism+preserve_weight*preservation_loss(result)-hardness_weight*hard
    tloss = task.loss(fake.detach(),batch,hardness=False)
    return dict(g=gloss,d=dloss,t=tloss,realism=realism,preserve=preservation_loss(result),
                hardness=hard,matched_regions=len(ids))
