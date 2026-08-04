_base_ = 'mmdet::rtmdet/rtmdet_s_8xb32-300e_coco.py'

data_root = '/home/katsuki/projects/face_detection/outsput/experience/data/mmdet_hf/'
work_dir = '/home/katsuki/projects/face_detection/outputs/experiments/rtmdet_s_head_mmdet_hf_960'
classes = ('head',)
metainfo = dict(classes=classes, palette=[(40, 220, 80)])
img_scale = (960, 960)

model = dict(
    backbone=dict(init_cfg=dict(
        type='Pretrained', prefix='backbone.',
        checkpoint='/home/katsuki/projects/face_detection/output_experiment/cache/torch/hub/checkpoints/cspnext-s_imagenet_600e.pth')),
    bbox_head=dict(num_classes=1),
    test_cfg=dict(score_thr=0.001, max_per_img=100))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=img_scale, pad_val=114.0, max_cached_images=20, random_pop=False),
    dict(type='RandomResize', scale=(1920, 1920), ratio_range=(0.5, 1.5), keep_ratio=True),
    dict(type='RandomCrop', crop_size=img_scale),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

train_dataloader = dict(batch_size=8, num_workers=4, persistent_workers=False, pin_memory=True, dataset=dict(
    type='CocoDataset', data_root=data_root, ann_file='train.json', data_prefix=dict(img=''),
    metainfo=metainfo, filter_cfg=dict(filter_empty_gt=False, min_size=1), pipeline=train_pipeline))
val_dataloader = dict(batch_size=8, num_workers=4, persistent_workers=False, pin_memory=True, drop_last=False,
                      sampler=dict(type='DefaultSampler', shuffle=False, round_up=False), dataset=dict(
                          type='CocoDataset', data_root=data_root, ann_file='val.json', data_prefix=dict(img=''),
                          metainfo=metainfo, test_mode=True, pipeline=test_pipeline))
test_dataloader = dict(batch_size=8, num_workers=4, persistent_workers=False, pin_memory=True, drop_last=False,
                       sampler=dict(type='DefaultSampler', shuffle=False, round_up=False), dataset=dict(
                           type='CocoDataset', data_root=data_root, ann_file='test.json', data_prefix=dict(img=''),
                           metainfo=metainfo, test_mode=True, pipeline=test_pipeline))
val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'val.json', metric='bbox', classwise=True)
test_evaluator = dict(type='CocoMetric', ann_file=data_root + 'test.json', metric='bbox', classwise=True)

max_epochs = 60
base_lr = 0.0002
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=2)
optim_wrapper = dict(
    _delete_=True, type='AmpOptimWrapper', loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=200),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.01, begin=0, end=max_epochs, T_max=max_epochs, by_epoch=True),
]
default_hooks = dict(
    checkpoint=dict(interval=2, save_best='coco/bbox_mAP_50', rule='greater', max_keep_ckpts=3, save_last=True),
    logger=dict(interval=20))
randomness = dict(seed=20260717, deterministic=False)
auto_scale_lr = dict(enable=False, base_batch_size=8)
