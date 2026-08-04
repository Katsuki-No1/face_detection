_base_ = ['mmpose::_base_/default_runtime.py']

custom_imports = dict(imports=['mmdet'], allow_failed_imports=False)
default_scope = 'mmpose'

num_keypoints = 5
input_size = (256, 256)
codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(5.66, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

max_epochs = 100
base_lr = 4e-3
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
randomness = dict(seed=0, deterministic=False)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.005,
        begin=30,
        end=max_epochs,
        T_max=max_epochs - 30,
        by_epoch=True,
        convert_to_iter_based=True),
]
auto_scale_lr = dict(base_batch_size=512)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.33,
        widen_factor=0.5,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint=(
                '/home/katsuki/projects/face_detection/output_experiment/'
                'cache/torch/hub/checkpoints/'
                'cspnext-s_imagenet_600e-ea671761.pth'))),
    head=dict(
        type='RTMCCHead',
        in_channels=512,
        out_channels=num_keypoints,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([size // 32 for size in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=False))

dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = '/home/katsuki/projects/face_detection/datasets/face_pose_drive_v1/'
annotation_root = (
    '/home/katsuki/projects/face_detection/outputs/experiments/'
    'rtmdet_s_face_pose_drive_v1_960_20260717/data/rtmpose_face/')
metainfo = dict(
    from_file=(
        '/home/katsuki/projects/face_detection/configs/rtmpose/'
        'face5_dataset_meta.py'))
backend_args = dict(backend='local')

train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomBBoxTransform', scale_factor=[0.75, 1.25], rotate_factor=20),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs'),
]

train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    # Avoid PyTorch 2.12 worker-shutdown hangs observed after completed runs.
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file=annotation_root + 'person_keypoints_train.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=False,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file=annotation_root + 'person_keypoints_val.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=False,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file=annotation_root + 'person_keypoints_test.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        test_mode=True,
        pipeline=val_pipeline))

val_evaluator = dict(
    type='NME', norm_mode='use_norm_item', norm_item='bbox_size')
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(
        save_best='NME',
        rule='less',
        max_keep_ckpts=3,
        interval=1,
        save_last=True),
    logger=dict(interval=20))

work_dir = (
    '/home/katsuki/projects/face_detection/outputs/experiments/'
    'rtmpose_s_face_pose_drive_v1_256_20260717')
