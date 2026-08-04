_base_ = './rtmpose_s_face_pose_drive_v1_256.py'

# The source ZIP remains immutable; training reads its verified cache.
dataset_version = 'face_pose_drive_v2_balanced_811'
data_root = 'outputs/dataset_cache/face_pose_current/'
annotation_root = (
    '{{fileDirname}}/../../outputs/dataset_adapters/'
    'face_pose_drive_v2_balanced_811/rtmpose/')
metainfo = dict(from_file='configs/rtmpose/face5_dataset_meta.py')
model = dict(
    backbone=dict(
        init_cfg=dict(
            checkpoint=(
                'output_experiment/cache/torch/hub/checkpoints/'
                'cspnext-s_imagenet_600e-ea671761.pth'))))

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=annotation_root + 'person_keypoints_train.json',
        data_prefix=dict(img=''),
        metainfo=metainfo))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=annotation_root + 'person_keypoints_val.json',
        data_prefix=dict(img=''),
        metainfo=metainfo))
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=annotation_root + 'person_keypoints_test.json',
        data_prefix=dict(img=''),
        metainfo=metainfo))

work_dir = 'outputs/experiments/rtmpose_s_face_pose_drive_v2_balanced_811_256'

dataset_contract = dict(
    dataset_version=dataset_version,
    images=dict(train=4237, val=530, test=528),
    annotations=dict(train=6529, val=840, test=769),
    coco_sha256=dict(
        train='4a8b21de648be724d38601c2a715adf27d1f5da9e42d7298edcfdd1d9102a6dc',
        val='6e60db201f9b90ecbf3c8fe92ed62f4e3cd624308441b21627c4e6c43950486a',
        test='8cb00ced078e73622056e1d639cece25f4ffd50cbfe4b607f38534478a5c9042'))
