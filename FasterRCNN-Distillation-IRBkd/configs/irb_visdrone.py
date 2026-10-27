# Adapted from MMDetection 2.28.2, Copyright (c) OpenMMLab, Apache-2.0.

custom_imports = dict(imports=['irb_kd'], allow_failed_imports=False)
model = dict(
    type='IRBFasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='DistillationRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='DistributionShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
            num_bins=17,
            d_max=2.0,
            dfl_loss_weight=0.25)),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))

import copy

teacher_model = copy.deepcopy(model)
teacher_model['backbone'].update(
    depth=101, init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101'))
model.update(
    teacher_model=teacher_model,
    teacher_checkpoint=None,
    distillation=dict(
        feature_distiller='irb-feature', logic_distiller='irb-logit',
        feature_layers=['backbone.layer1', 'backbone.layer2',
                        'backbone.layer3', 'backbone.layer4'],
        channels_s=[256, 512, 1024, 2048], channels_t=[256, 512, 1024, 2048],
        alpha=4e-5, beta=0.8, gamma=6e-5,
        relation_temperature=0.5, boundary_temperature=0.4,
        boundary_criterion='w1', rho_top=0.2, rho_tail=0.2,
        roi_size=7, embed_dim=128, schedule='constant', class_counts=None))
del teacher_model, copy

classes = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
           'tricycle', 'awning-tricycle', 'bus', 'motor')
data_root = 'data/VisDrone/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug', img_scale=(1333, 800), flip=False, transforms=[
        dict(type='Resize', keep_ratio=True), dict(type='RandomFlip'),
        dict(type='Normalize', **img_norm_cfg), dict(type='Pad', size_divisor=32),
        dict(type='ImageToTensor', keys=['img']), dict(type='Collect', keys=['img'])])]
data = dict(
    samples_per_gpu=2, workers_per_gpu=2,
    train=dict(type='CocoDataset', data_root=data_root, classes=classes,
               ann_file='annotations/train.json', img_prefix='VisDrone2019-DET-train/images/',
               filter_empty_gt=False, pipeline=train_pipeline),
    val=dict(type='CocoDataset', data_root=data_root, classes=classes,
             ann_file='annotations/val.json', img_prefix='VisDrone2019-DET-val/images/', pipeline=test_pipeline),
    test=dict(type='CocoDataset', data_root=data_root, classes=classes,
              ann_file='annotations/test-dev.json', img_prefix='VisDrone2019-DET-test-dev/images/', pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP', rule='greater',
                  metric_items=['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'])

optimizer = dict(type='SGD', constructor='TrainableOptimizerConstructor',
                 lr=0.01, momentum=0.937, weight_decay=0.0005)
optimizer_config = dict(type='FiniteOptimizerHook', grad_clip=None)
lr_config = dict(policy='step', warmup='linear', warmup_iters=500,
                 warmup_ratio=0.001, step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
find_unused_parameters = False
auto_scale_lr = dict(enable=False, base_batch_size=8)
custom_hooks = [dict(type='IterationStateHook', priority='HIGH')]

seed = 0
