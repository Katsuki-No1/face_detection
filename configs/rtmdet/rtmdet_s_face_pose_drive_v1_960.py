_base_ = 'mmdet::rtmdet/rtmdet_s_8xb32-300e_coco.py'

# RTMDet-S face-only baseline on the fixed face_pose_drive_v1 group split.
# This mirrors the 3,090-frame benchmark protocol: 960 px, batch 16, seed 0,
# validation-only checkpoint selection, and an untouched independent test split.

data_root = '/home/katsuki/projects/face_detection/datasets/face_pose_drive_v1/'
work_dir = '/home/katsuki/projects/face_detection/outputs/experiments/rtmdet_s_face_pose_drive_v1_960_low_lr_20260717'

classes = ('face', )
metainfo = dict(classes=classes, palette=[(80, 160, 255)])
num_classes = 1
img_scale = (960, 960)

model = dict(
    bbox_head=dict(num_classes=num_classes),
    test_cfg=dict(score_thr=0.001, max_per_img=100))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='CachedMosaic',
        img_scale=img_scale,
        pad_val=114.0,
        max_cached_images=20,
        random_pop=False),
    dict(
        type='RandomResize',
        scale=(1920, 1920),
        ratio_range=(0.5, 1.5),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=img_scale),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs'),
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=img_scale,
        ratio_range=(0.75, 1.25),
        keep_ratio=True),
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
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    # Avoid PyTorch 2.12 worker-shutdown hangs observed after completed runs.
    persistent_workers=False,
    pin_memory=True,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=False,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=False,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric='bbox',
    classwise=True,
    proposal_nums=(100, 1, 10))
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test.json',
    metric='bbox',
    classwise=True,
    proposal_nums=(100, 1, 10))

# The initial 1e-3 run peaked at epoch 5 and then destabilized.  Use a
# fine-tuning learning rate and a shorter schedule for the fixed 3,090-frame
# dataset; checkpoint selection still uses validation only.
max_epochs = 60
stage2_num_epochs = 10
base_lr = 0.0002

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.01,
        begin=0,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True),
]

default_hooks = dict(
    checkpoint=dict(
        interval=5,
        save_best='coco/bbox_mAP',
        rule='greater',
        max_keep_ckpts=5,
        save_last=True),
    logger=dict(interval=20))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2),
]

randomness = dict(seed=0, deterministic=False)
auto_scale_lr = dict(enable=False, base_batch_size=16)

# Initialize the detector from the official RTMDet-S COCO checkpoint. The
# one-class prediction layers are intentionally reinitialized on shape mismatch.
load_from = ('https://download.openmmlab.com/mmdetection/v3.0/rtmdet/'
             'rtmdet_s_8xb32-300e_coco/'
             'rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth')
