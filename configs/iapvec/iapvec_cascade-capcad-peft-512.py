_base_ = ['_base_/iapvec_cascade.py']

work_dir = './work_dirs/iapvec/iapvec_cascade-capcad-peft'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=500),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, 
                    #max_keep_ckpts=6, 
                    save_best='coco/segm_mAP', rule='greater', save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # visualization=dict(type='DetVisualizationHook', draw=True, interval=1, test_out_dir='vis_data')
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

num_classes = 1
prompt_shape = (100, 5)  # (per img pointset, per pointset point)

#### should be changed when using different pretrain model

# sam base model
# change align with your weight root
hf_sam_pretrain_name = "/home/data4/zar24/code/RSPrompter/pretrained/sam-base"
hf_sam_pretrain_ckpt_path = "/home/data4/zar24/code/RSPrompter/pretrained/sam-base/pytorch_model.bin"

crop_size = (512, 512)

batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=crop_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False)
]

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    bgr_to_rgb=True,
    pad_mask=True,
    pad_size_divisor=32,
    batch_augments=batch_augments
)

model = dict(
    decoder_freeze=False,
    post_process = False, #False when training
    data_preprocessor=data_preprocessor,
    shared_image_embedding=dict(
        hf_pretrain_name=hf_sam_pretrain_name,
        init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path),
    ),
    backbone=dict(
        _delete_=True,
        img_size=crop_size[0],
        type='MMPretrainSamVisionEncoder',
        hf_pretrain_name=hf_sam_pretrain_name,
        init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path),
        peft_config=dict(
            peft_type="LORA",
            r=16,
            target_modules=["qkv"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
        ),
    ),
    neck=dict(
        feature_aggregator=dict(
            _delete_=True,
            type='PseudoFeatureAggregator',
            in_channels=256,
            hidden_channels=512,
            out_channels=256,
        ),
    ),
)


backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args, to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    # large scale jittering
    dict(
        type='Resize',
        scale=crop_size,
        keep_ratio=True
    ),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args, to_float32=True),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='Pad', size=crop_size, pad_val=dict(img=(0.406 * 255, 0.456 * 255, 0.485 * 255), masks=0)),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor'))
]

dataset_type = 'CAPCADInsSegDataset'
#### should be changed align with your code root and data root
code_root = '/home/data4/zar24/code/RSPrompter-open'
data_root = '/home/data4/zar24/data/seven_region'

batch_size_per_gpu = 2
num_workers = 4
persistent_workers = True
train_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root+'/annotations/train.json',
        data_prefix=dict(img='dataset/train/images/'),
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=persistent_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/annotations/test.json',
        data_prefix=dict(img='dataset/test/images/'),
        pipeline=test_pipeline,
    )
)

test_dataloader = val_dataloader

test_evaluator = dict(
         type='CocoMetric',
         metric=['bbox', 'segm'],
         format_only=False,
         )  #output json file

resume = False
load_from = None

base_lr = 0.0001
max_epochs = 200

train_cfg = dict(max_epochs=max_epochs)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=50),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.001,
        begin=1,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True
    )
]

### AMP training config
runner_type = 'Runner'
optim_wrapper = dict(
     type='AmpOptimWrapper',
     dtype='float16',
     optimizer=dict(
         type='AdamW',
         lr=base_lr,
         weight_decay=0.05)
 )
