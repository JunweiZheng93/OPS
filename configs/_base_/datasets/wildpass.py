# dataloaders
val_dataloader = dict(batch_size=1,  # batch size per GPU
                      num_workers=4,  # number of workers to load data
                      persistent_workers=True,
                      pin_memory=True,
                      drop_last=True,
                      dataset=dict(type='WildPassDataset',
                                   data_root='data/WildPASS',
                                   data_prefix=dict(img_path='images/val', seg_map_path='annotations/val'),
                                   img_suffix='.png',  # suffix of images (there might be other files in the image directory, so we only read files ending with 'img_suffix'). also, img_suffix will be replaced by seg_map_suffix to get the segmentation map path
                                   seg_map_suffix='_labelTrainIds.png',  # suffix of segmentation maps (there might be other files in the segmentation map directory, so we only read files ending with 'seg_map_suffix')
                                   ignore_index=255,  # the index of the label to ignore in the segmentation map when calculating loss
                                   reduce_zero_label=False,  # some datasets use 0 as the ignore_index (wildpass use 255, so we set it to False)
                                   pipeline=[dict(type='LoadImageFromFile'),
                                             dict(type='Resize', scale=(2048, 512), keep_ratio=True),
                                             dict(type='LoadAnnotations'),
                                             dict(type='PackSegInputs')]),
                      sampler=dict(type='DefaultSampler',  # DefaultSampler is designed for epoch-based training. It can handle both distributed and non-distributed training.
                                   shuffle=False),
                      collate_fn=dict(type='default_collate'))  # this will concatenate all the data in a batch into a single tensor

test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator