# ---------------------------------------------------------------
#  Modified Config: SFSCNet + Faster R-CNN + COCO Format (SSDD)
# ---------------------------------------------------------------

# 1. 基础配置继承：将 voc0712 改为 coco_detection
_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py', # <--- 修改这里
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='SFSCNet',   
        base_channels=64, 
        arch_settings=[2, 2, 2, 2], 
        filter_type='FrGT', 
        out_indices=(0, 1, 2, 3), 
        init_cfg=dict(type='Kaiming', layer='Conv2d')
    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512], # 假设 SFS-CNet 对应输出
        out_channels=256,
        num_outs=5
    ),
    roi_head=dict(
        bbox_head=dict(
            type='ConvFCBBoxHead',
            num_shared_convs=4,
            num_shared_fcs=1,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
    ),
)

# 3. 数据集设置 (Dataset Settings)
dataset_type = 'CocoDataset'  # <--- 修改类型
data_root = '/home/user/4T_Storage/SJY/SAR/SARATR-X/detection/dataset/SSDD/'

# 非常重要：显式定义元数据，覆盖 COCO 默认的 80 类
metainfo = dict(
    classes=('ship', ),
    palette=[(220, 20, 60)]
)

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(608, 608), keep_ratio=True), # 修改为常用尺寸
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(608, 608), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# Dataloader 配置
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True, # 删除继承自 base 的配置，完全重写
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo, # 注入类别信息
        ann_file='annotations/train.json', # 指向 JSON 文件
        data_prefix=dict(img='images/train/'), # COCO 格式通常使用 'img' 键
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/test/'),
        test_mode=True,
        pipeline=test_pipeline
    )
)
test_dataloader = val_dataloader

# 4. 评估指标 (Evaluator) - 从 VOCMetric 改为 CocoMetric
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test.json',
    metric='bbox',
    format_only=False,
    backend_args=None
)
test_evaluator = val_evaluator

# 5. 训练策略设置 (Schedule & Runtime)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1
    )
]

visualizer = dict(
    type='DetLocalVisualizer', 
    vis_backends=[dict(type='LocalVisBackend')], 
    name='visualizer'
)