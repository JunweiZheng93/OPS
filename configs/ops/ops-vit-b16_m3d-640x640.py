_base_ = [
    '../_base_/models/ops_vit-b16.py', '../_base_/datasets/m3d.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

crop_size = (640, 640)
pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/san/clip_vit-base-patch16-224_3rdparty-d08f8887.pth'  # noqa
data_preprocessor = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    size_divisor=640,
    test_cfg=dict(size_divisor=32))
model = dict(
    pretrained=pretrained,
    text_encoder=dict(dataset_name=None,
                      vocabulary=['wall', 'floor', 'chair', 'door', 'table', 'picture', 'furniture', 'objects', 'window', 'sofa', 'bed', 'sink', 'stairs', 'ceiling', 'toilet', 'mirror', 'shower', 'bathtub', 'counter', 'shelving']),
    decode_head=dict(num_classes=20, ops_cfg=dict(cfg_dao=dict(dw_kernel_size=3))),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(320, 320))
)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeShortestEdge', scale=crop_size, max_size=2560),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
val_dataloader = dict(batch_size=1, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
default_hooks = dict(visualization=dict(type='SegVisualizationHook', draw=False, interval=1))



# ==================== not important, since we don't train the model ====================

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomChoiceResize',
        scales=[int(640 * x * 0.1) for x in range(5, 16)],
        resize_type='ResizeShortestEdge',
        max_size=2560),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

train_dataloader = dict(batch_size=8,  # batch size per GPU
                        num_workers=4,  # number of workers to load data
                        persistent_workers=True,
                        pin_memory=True,
                        drop_last=True,
                        dataset=dict(type='M3DDataset',
                                     data_root='data/matterport3d',
                                     data_prefix=dict(img_path='val/rgb', seg_map_path='val/semantic'),
                                     img_suffix='.png',  # suffix of images (there might be other files in the image directory, so we only read files ending with 'img_suffix'). also, img_suffix will be replaced by seg_map_suffix to get the segmentation map path
                                     seg_map_suffix='.png',  # suffix of segmentation maps (there might be other files in the segmentation map directory, so we only read files ending with 'seg_map_suffix')
                                     ignore_index=255,  # the index of the label to ignore in the segmentation map when calculating loss
                                     reduce_zero_label=True,  # some datasets use 0 as the ignore_index (wildpass use 255, so we set it to False)
                                     pipeline=train_pipeline),
                        sampler=dict(type='DefaultSampler',  # DefaultSampler is designed for epoch-based training. It can handle both distributed and non-distributed training.
                                     shuffle=False),
                        collate_fn=dict(type='default_collate'))  # this will concatenate all the data in a batch into a single tensor

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=60000,
    val_interval=500,
    val_begin=55000)

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'img_encoder': dict(lr_mult=0.1, decay_mult=1.0),
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    loss_scale='dynamic',
    clip_grad=dict(max_norm=0.01, norm_type=2))

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=0,
        end=60000,
        by_epoch=False,
    )
]