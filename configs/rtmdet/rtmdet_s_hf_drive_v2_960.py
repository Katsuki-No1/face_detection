_base_ = './rtmdet_s_hf_full_960_wandb.py'

# The source ZIP remains immutable. Images and bundled Head/Face COCO files
# are read from the verified cache selected by default_face_pose.yaml.
dataset_version = 'face_pose_drive_v2_balanced_811_head_face'
data_root = 'outputs/dataset_cache/face_pose_current/'
annotation_root = 'annotations/rtmdet_hf/'
work_dir = 'outputs/experiments/rtmdet_s_hf_drive_v2_balanced_811_960'
max_epochs = 60
base_lr = 0.0002

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=annotation_root + 'train.json',
        data_prefix=dict(img='')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=annotation_root + 'val.json',
        data_prefix=dict(img='')))
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=annotation_root + 'test.json',
        data_prefix=dict(img='')))

val_evaluator = dict(
    ann_file=data_root + annotation_root + 'val.json')
test_evaluator = dict(
    ann_file=data_root + annotation_root + 'test.json')

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        save_dir=work_dir,
        init_kwargs=dict(
            entity='sato-waseda-university',
            project='face_detection',
            group='rtmdet_hf_production',
            name='rtmdet_s_hf_drive_v2_balanced_811_960_seed42',
            id='rtmdet-s-hf-drive-v2-balanced-811-960-s42',
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

dataset_contract = dict(
    dataset_version=dataset_version,
    images=dict(train=4237, val=530, test=528),
    annotations=dict(train=13658, val=1722, test=1575),
    category_annotations=dict(
        train=dict(head=7129, face=6529),
        val=dict(head=882, face=840),
        test=dict(head=806, face=769)),
    coco_sha256=dict(
        train='25f3e4c7ed3a96c250a0bafc28138504f0295ecbcba195c87d4868307534086f',
        val='d13cdf7ca12e181a8d6f9550eb87975b240ebb1c8a29fa83413d1da573053311',
        test='e64fa7c624a87750306b93a6be8f363bf1134c2f27277b2efbf14b7cd45af7be'))
