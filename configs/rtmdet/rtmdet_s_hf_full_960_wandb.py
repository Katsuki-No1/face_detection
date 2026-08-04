_base_ = 'mmdet::rtmdet/rtmdet_s_8xb32-300e_coco.py'

# Two-class RTMDet-HF training on the 3,090-frame dataset converted directly
# from face_pose_drive_v1 metadata. W&B records the run in the shared project.
dataset_version = 'rtmdet_hf_drive_v1'
data_root = 'datasets/rtmdet_hf_drive_v1/'
work_dir = 'outputs/experiments/rtmdet_s_hf_drive_v1_960_low_lr_20260719'

classes = ('head', 'face')
metainfo = dict(classes=classes, palette=[(40, 220, 80), (80, 160, 255)])
num_classes = 2
img_scale = (960, 960)
max_epochs = 60
stage2_num_epochs = 10
base_lr = 0.0002

model = dict(
    data_preprocessor=dict(pad_size_divisor=32),
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
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
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

val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'val.json', metric='bbox', classwise=True)
test_evaluator = dict(type='CocoMetric', ann_file=data_root + 'test.json', metric='bbox', classwise=True)

train_cfg = dict(
    _delete_=True,
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
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True))
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
        max_keep_ckpts=3,
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
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        rule='greater',
        min_delta=0.0005,
        patience=15,
        strict=True),
]

randomness = dict(seed=42, deterministic=False)
auto_scale_lr = dict(enable=False, base_batch_size=16)

load_from = (
    'output_experiment/cache/torch/hub/checkpoints/'
    'rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth')

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        save_dir=work_dir,
        init_kwargs=dict(
            entity='sato-waseda-university',
            project='face_detection',
            group='rtmdet_hf_production',
            name='rtmdet_s_hf_drive_v1_960_low_lr_seed42_20260719',
            id='rtmdet-s-hf-drive-v1-960-low-lr-s42-20260719',
            resume='allow',
            job_type='training',
            tags=['RTMDet', 'head', 'face', 'production', dataset_version],
            config=dict(
                model_family='RTMDet',
                model_size='s',
                dataset_version=dataset_version,
                epochs=max_epochs,
                batch_size=16,
                imgsz=960,
                optimizer='AdamW',
                learning_rate=base_lr,
                cosine_begin_epoch=0,
                cosine_end_epoch=max_epochs,
                amp=True,
                ema=True,
                mosaic=True,
                seed=42)))
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

# Dataset contract used for the preflight/report. MMDetection ignores this
# metadata, but it makes accidental dataset replacement detectable.
dataset_contract = dict(
    dataset_version=dataset_version,
    images=dict(train=2460, val=300, test=330),
    annotations=dict(train=8563, val=1028, test=1015),
    category_annotations=dict(
        train=dict(head=4536, face=4027),
        val=dict(head=525, face=503),
        test=dict(head=512, face=503)),
    coco_sha256=dict(
        train='f52b2cac17c02ef5415758155cd0c2ad26a5e6f39cd9685a0d6d59dc0201d7b8',
        val='c027fc20aa67ac17a018831a8572bf830480eb11708c657fae589adebfd33414',
        test='93d644f3aa6c97c6c5cc12e8a6fc48b560d1499d87bcc8f8c3dc8ad41c5e1369'))
