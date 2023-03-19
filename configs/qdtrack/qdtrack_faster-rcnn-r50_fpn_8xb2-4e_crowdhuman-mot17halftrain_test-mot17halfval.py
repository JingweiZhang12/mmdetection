_base_ = [
    './qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py'
]

# data pipeline
train_pipeline = [
    dict(
        type='UniformSample',
        num_ref_imgs=1,
        frame_range=10,
        filter_key_img=True),
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadTrackAnnotations', with_instance_id=True),
            dict(
                type='RandomResize',
                scale=(1088, 1088),
                ratio_range=(0.8, 1.2),
                keep_ratio=True,
                clip_object_border=False),
            dict(type='PhotoMetricDistortion')
        ]),
    dict(
        type='TransformBroadcaster',
        share_random_params=False,
        transforms=[
            dict(
                type='RandomCrop',
                crop_size=(1088, 1088),
                bbox_clip_border=False)
        ]),
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='RandomFlip', prob=0.5),
        ]),
    dict(type='PackTrackInputs')
]
mot_cfg = dict(
    type='MOTChallengeDataset',
    data_root='data/MOT17',
    visibility_thr=-1,
    ann_file='annotations/half-train_cocoformat.json',
    data_prefix=dict(img_path='train'),
    metainfo=dict(classes=('pedestrian')),
    pipeline=train_pipeline)
crowdhuman_cfg = dict(
    type='BaseVideoDataset',
    data_root='data/crowdhuman',
    metainfo=dict(classes=('pedestrian')),
    ann_file='annotations/crowdhuman_train.json',
    data_prefix=dict(img_path='train'),
    pipeline=train_pipeline)

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='ImgQuotaSampler'),
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[mot_cfg, crowdhuman_cfg]))
