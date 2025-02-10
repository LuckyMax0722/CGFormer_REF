data_root = 'data/semantickitti'
ann_file = 'data/semantickitti/labels'
stereo_depth_root = 'data/semantickitti/depth'
camera_used = ['left']
dataset_type = 'SemanticKITTIDataset'
point_cloud_range = [0, -25.6, -2, 51.2, 25.6, 4.4]
occ_size = [256, 256, 32]
semantic_kitti_class_frequencies = [
    5417730330.0, 15783539.0, 125136.0, 118809.0, 646799.0, 821951.0, 262978.0,
    283696.0, 204750.0, 61688703.0, 4502961.0, 44883650.0, 2269923.0,
    56840218.0, 15719652.0, 158442623.0, 2061623.0, 36970522.0, 1151988.0,
    334146.0
]
class_names = [
    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
    'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
    'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
    'pole', 'traffic-sign'
]
num_class = 20
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    flip_dz_ratio=0)
data_config = dict(
    input_size=(384, 1280),
    resize=(0.0, 0.0),
    rot=(0.0, 0.0),
    flip=False,
    crop_h=(0.0, 0.0),
    resize_test=0.0)
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        data_config=dict(
            input_size=(384, 1280),
            resize=(0.0, 0.0),
            rot=(0.0, 0.0),
            flip=False,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        load_stereo_depth=True,
        is_train=True,
        color_jitter=(0.4, 0.4, 0.4)),
    dict(
        type='CreateDepthFromLiDAR',
        data_root='data/semantickitti',
        dataset='kitti',
        load_seg=False),
    dict(
        type='LoadAnnotationOcc',
        bda_aug_conf=dict(
            rot_lim=(-22.5, 22.5),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5,
            flip_dz_ratio=0),
        apply_bda=False,
        is_train=True,
        point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
    dict(
        type='CollectData',
        keys=['img_inputs', 'gt_occ'],
        meta_keys=[
            'pc_range', 'occ_size', 'raw_img', 'stereo_depth', 'focal_length',
            'baseline', 'img_shape', 'gt_depths'
        ])
]
trainset_config = dict(
    type='SemanticKITTIDataset',
    stereo_depth_root='data/semantickitti/depth',
    data_root='data/semantickitti',
    ann_file='data/semantickitti/labels',
    pipeline=[
        dict(
            type='LoadMultiViewImageFromFiles',
            data_config=dict(
                input_size=(384, 1280),
                resize=(0.0, 0.0),
                rot=(0.0, 0.0),
                flip=False,
                crop_h=(0.0, 0.0),
                resize_test=0.0),
            load_stereo_depth=True,
            is_train=True,
            color_jitter=(0.4, 0.4, 0.4)),
        dict(
            type='CreateDepthFromLiDAR',
            data_root='data/semantickitti',
            dataset='kitti',
            load_seg=False),
        dict(
            type='LoadAnnotationOcc',
            bda_aug_conf=dict(
                rot_lim=(-22.5, 22.5),
                scale_lim=(0.95, 1.05),
                flip_dx_ratio=0.5,
                flip_dy_ratio=0.5,
                flip_dz_ratio=0),
            apply_bda=False,
            is_train=True,
            point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
        dict(
            type='CollectData',
            keys=['img_inputs', 'gt_occ'],
            meta_keys=[
                'pc_range', 'occ_size', 'raw_img', 'stereo_depth',
                'focal_length', 'baseline', 'img_shape', 'gt_depths'
            ])
    ],
    split='train',
    camera_used=['left'],
    occ_size=[256, 256, 32],
    pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
    test_mode=False)
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        data_config=dict(
            input_size=(384, 1280),
            resize=(0.0, 0.0),
            rot=(0.0, 0.0),
            flip=False,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        load_stereo_depth=True,
        is_train=False,
        color_jitter=None),
    dict(
        type='CreateDepthFromLiDAR',
        data_root='data/semantickitti',
        dataset='kitti'),
    dict(
        type='LoadAnnotationOcc',
        bda_aug_conf=dict(
            rot_lim=(-22.5, 22.5),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5,
            flip_dz_ratio=0),
        apply_bda=False,
        is_train=False,
        point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
    dict(
        type='CollectData',
        keys=['img_inputs', 'gt_occ'],
        meta_keys=[
            'pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img',
            'stereo_depth', 'focal_length', 'baseline', 'img_shape',
            'gt_depths'
        ])
]
testset_config = dict(
    type='SemanticKITTIDataset',
    stereo_depth_root='data/semantickitti/depth',
    data_root='data/semantickitti',
    ann_file='data/semantickitti/labels',
    pipeline=[
        dict(
            type='LoadMultiViewImageFromFiles',
            data_config=dict(
                input_size=(384, 1280),
                resize=(0.0, 0.0),
                rot=(0.0, 0.0),
                flip=False,
                crop_h=(0.0, 0.0),
                resize_test=0.0),
            load_stereo_depth=True,
            is_train=False,
            color_jitter=None),
        dict(
            type='CreateDepthFromLiDAR',
            data_root='data/semantickitti',
            dataset='kitti'),
        dict(
            type='LoadAnnotationOcc',
            bda_aug_conf=dict(
                rot_lim=(-22.5, 22.5),
                scale_lim=(0.95, 1.05),
                flip_dx_ratio=0.5,
                flip_dy_ratio=0.5,
                flip_dz_ratio=0),
            apply_bda=False,
            is_train=False,
            point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
        dict(
            type='CollectData',
            keys=['img_inputs', 'gt_occ'],
            meta_keys=[
                'pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img',
                'stereo_depth', 'focal_length', 'baseline', 'img_shape',
                'gt_depths'
            ])
    ],
    split='test',
    camera_used=['left'],
    occ_size=[256, 256, 32],
    pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4])
data = dict(
    train=dict(
        type='SemanticKITTIDataset',
        stereo_depth_root='data/semantickitti/depth',
        data_root='data/semantickitti',
        ann_file='data/semantickitti/labels',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles',
                data_config=dict(
                    input_size=(384, 1280),
                    resize=(0.0, 0.0),
                    rot=(0.0, 0.0),
                    flip=False,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                load_stereo_depth=True,
                is_train=True,
                color_jitter=(0.4, 0.4, 0.4)),
            dict(
                type='CreateDepthFromLiDAR',
                data_root='data/semantickitti',
                dataset='kitti',
                load_seg=False),
            dict(
                type='LoadAnnotationOcc',
                bda_aug_conf=dict(
                    rot_lim=(-22.5, 22.5),
                    scale_lim=(0.95, 1.05),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5,
                    flip_dz_ratio=0),
                apply_bda=False,
                is_train=True,
                point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
            dict(
                type='CollectData',
                keys=['img_inputs', 'gt_occ'],
                meta_keys=[
                    'pc_range', 'occ_size', 'raw_img', 'stereo_depth',
                    'focal_length', 'baseline', 'img_shape', 'gt_depths'
                ])
        ],
        split='train',
        camera_used=['left'],
        occ_size=[256, 256, 32],
        pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
        test_mode=False),
    val=dict(
        type='SemanticKITTIDataset',
        stereo_depth_root='data/semantickitti/depth',
        data_root='data/semantickitti',
        ann_file='data/semantickitti/labels',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles',
                data_config=dict(
                    input_size=(384, 1280),
                    resize=(0.0, 0.0),
                    rot=(0.0, 0.0),
                    flip=False,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                load_stereo_depth=True,
                is_train=False,
                color_jitter=None),
            dict(
                type='CreateDepthFromLiDAR',
                data_root='data/semantickitti',
                dataset='kitti'),
            dict(
                type='LoadAnnotationOcc',
                bda_aug_conf=dict(
                    rot_lim=(-22.5, 22.5),
                    scale_lim=(0.95, 1.05),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5,
                    flip_dz_ratio=0),
                apply_bda=False,
                is_train=False,
                point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
            dict(
                type='CollectData',
                keys=['img_inputs', 'gt_occ'],
                meta_keys=[
                    'pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img',
                    'stereo_depth', 'focal_length', 'baseline', 'img_shape',
                    'gt_depths'
                ])
        ],
        split='test',
        camera_used=['left'],
        occ_size=[256, 256, 32],
        pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
    test=dict(
        type='SemanticKITTIDataset',
        stereo_depth_root='data/semantickitti/depth',
        data_root='data/semantickitti',
        ann_file='data/semantickitti/labels',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles',
                data_config=dict(
                    input_size=(384, 1280),
                    resize=(0.0, 0.0),
                    rot=(0.0, 0.0),
                    flip=False,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                load_stereo_depth=True,
                is_train=False,
                color_jitter=None),
            dict(
                type='CreateDepthFromLiDAR',
                data_root='data/semantickitti',
                dataset='kitti'),
            dict(
                type='LoadAnnotationOcc',
                bda_aug_conf=dict(
                    rot_lim=(-22.5, 22.5),
                    scale_lim=(0.95, 1.05),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5,
                    flip_dz_ratio=0),
                apply_bda=False,
                is_train=False,
                point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
            dict(
                type='CollectData',
                keys=['img_inputs', 'gt_occ'],
                meta_keys=[
                    'pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img',
                    'stereo_depth', 'focal_length', 'baseline', 'img_shape',
                    'gt_depths'
                ])
        ],
        split='test',
        camera_used=['left'],
        occ_size=[256, 256, 32],
        pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4]))
train_dataloader_config = dict(batch_size=1, num_workers=4)
test_dataloader_config = dict(batch_size=1, num_workers=4)
numC_Trans = 128
lss_downsample = [2, 2, 2]
voxel_out_channels = [128]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
voxel_x = 0.2
voxel_y = 0.2
voxel_z = 0.2
grid_config = dict(
    xbound=[0, 51.2, 0.4],
    ybound=[-25.6, 25.6, 0.4],
    zbound=[-2, 4.4, 0.4],
    dbound=[2.0, 58.0, 0.5])
_num_layers_cross_ = 3
_num_points_cross_ = 8
_num_levels_ = 1
_num_cams_ = 1
_dim_ = 128
_pos_dim_ = 64
_num_layers_self_ = 2
_num_points_self_ = 8
model = dict(
    type='CGFormer',
    img_backbone=dict(
        type='CustomEfficientNet',
        arch='b7',
        drop_path_rate=0.2,
        frozen_stages=0,
        norm_eval=False,
        out_indices=(2, 3, 4, 5, 6),
        with_cp=True,
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone',
            checkpoint=
            './ckpts/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth'
        )),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[48, 80, 224, 640, 2560],
        upsample_strides=[0.5, 1, 2, 4, 4],
        out_channels=[128, 128, 128, 128, 128]),
    depth_net=dict(
        type='GeometryDepth_Net',
        downsample=8,
        numC_input=640,
        numC_Trans=128,
        cam_channels=33,
        grid_config=dict(
            xbound=[0, 51.2, 0.4],
            ybound=[-25.6, 25.6, 0.4],
            zbound=[-2, 4.4, 0.4],
            dbound=[2.0, 58.0, 0.5]),
        loss_depth_type='kld',
        loss_depth_weight=0.0001),
    img_view_transformer=dict(
        type='LSSViewTransformer',
        downsample=8,
        grid_config=dict(
            xbound=[0, 51.2, 0.4],
            ybound=[-25.6, 25.6, 0.4],
            zbound=[-2, 4.4, 0.4],
            dbound=[2.0, 58.0, 0.5]),
        data_config=dict(
            input_size=(384, 1280),
            resize=(0.0, 0.0),
            rot=(0.0, 0.0),
            flip=False,
            crop_h=(0.0, 0.0),
            resize_test=0.0)),
    proposal_layer=dict(
        type='VoxelProposalLayer',
        point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
        input_dimensions=[128, 128, 16],
        data_config=dict(
            input_size=(384, 1280),
            resize=(0.0, 0.0),
            rot=(0.0, 0.0),
            flip=False,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        init_cfg=None),
    VoxFormer_head=dict(
        type='VoxFormerHead',
        volume_h=128,
        volume_w=128,
        volume_z=16,
        data_config=dict(
            input_size=(384, 1280),
            resize=(0.0, 0.0),
            rot=(0.0, 0.0),
            flip=False,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
        embed_dims=128,
        cross_transformer=dict(
            type='PerceptionTransformer_DFA3D',
            rotate_prev_bev=True,
            use_shift=True,
            embed_dims=128,
            num_cams=1,
            encoder=dict(
                type='VoxFormerEncoder_DFA3D',
                num_layers=3,
                pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
                data_config=dict(
                    input_size=(384, 1280),
                    resize=(0.0, 0.0),
                    rot=(0.0, 0.0),
                    flip=False,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                num_points_in_pillar=8,
                return_intermediate=False,
                transformerlayers=dict(
                    type='VoxFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='DeformCrossAttention_DFA3D',
                            pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
                            num_cams=1,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D_DFA3D',
                                embed_dims=128,
                                num_points=8,
                                num_levels=1),
                            embed_dims=128)
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=128,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    feedforward_channels=256,
                    ffn_dropout=0.1,
                    operation_order=('cross_attn', 'norm', 'ffn', 'norm')))),
        self_transformer=dict(
            type='PerceptionTransformer_DFA3D',
            rotate_prev_bev=True,
            use_shift=True,
            embed_dims=128,
            num_cams=1,
            use_level_embeds=False,
            use_cams_embeds=False,
            encoder=dict(
                type='VoxFormerEncoder',
                num_layers=2,
                pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
                data_config=dict(
                    input_size=(384, 1280),
                    resize=(0.0, 0.0),
                    rot=(0.0, 0.0),
                    flip=False,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                num_points_in_pillar=8,
                return_intermediate=False,
                transformerlayers=dict(
                    type='VoxFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='DeformSelfAttention',
                            embed_dims=128,
                            num_levels=1,
                            num_points=8)
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=128,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    feedforward_channels=256,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')))),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=64,
            row_num_embed=512,
            col_num_embed=512),
        mlp_prior=True),
    occ_encoder_backbone=dict(
        type='Fuser',
        embed_dims=128,
        global_aggregator=dict(
            type='TPVGlobalAggregator',
            embed_dims=128,
            split=[8, 8, 8],
            grid_size=[128, 128, 16],
            global_encoder_backbone=dict(
                type='Swin',
                embed_dims=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                in_channels=128,
                patch_size=4,
                strides=[1, 2, 2, 2],
                frozen_stages=-1,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.2,
                patch_norm=True,
                out_indices=[1, 2, 3],
                with_cp=False,
                convert_weights=True,
                init_cfg=dict(
                    type='Pretrained',
                    checkpoint='./ckpts/swin_tiny_patch4_window7_224.pth')),
            global_encoder_neck=dict(
                type='GeneralizedLSSFPN',
                in_channels=[192, 384, 768],
                out_channels=128,
                start_level=0,
                num_outs=3,
                norm_cfg=dict(
                    type='BN2d', requires_grad=True,
                    track_running_stats=False),
                act_cfg=dict(type='ReLU', inplace=True),
                upsample_cfg=dict(mode='bilinear', align_corners=False))),
        local_aggregator=dict(
            type='LocalAggregator',
            local_encoder_backbone=dict(
                type='CustomResNet3D',
                numC_input=128,
                num_layer=[2, 2, 2],
                num_channels=[128, 128, 128],
                stride=[1, 2, 2]),
            local_encoder_neck=dict(
                type='GeneralizedLSSFPN',
                in_channels=[128, 128, 128],
                out_channels=128,
                start_level=0,
                num_outs=3,
                norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                conv_cfg=dict(type='Conv3d'),
                act_cfg=dict(type='ReLU', inplace=True),
                upsample_cfg=dict(mode='trilinear', align_corners=False)))),
    pts_bbox_head=dict(
        type='OccHead',
        in_channels=[128],
        out_channel=20,
        empty_idx=0,
        num_level=1,
        with_cp=True,
        occ_size=[256, 256, 32],
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0),
        conv_cfg=dict(type='Conv3d', bias=False),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        class_frequencies=[
            5417730330.0, 15783539.0, 125136.0, 118809.0, 646799.0, 821951.0,
            262978.0, 283696.0, 204750.0, 61688703.0, 4502961.0, 44883650.0,
            2269923.0, 56840218.0, 15719652.0, 158442623.0, 2061623.0,
            36970522.0, 1151988.0, 334146.0
        ]))
learning_rate = 0.0003
training_steps = 25000
optimizer = dict(type='AdamW', lr=0.0003, weight_decay=0.01)
lr_scheduler = dict(
    type='OneCycleLR',
    max_lr=0.0003,
    total_steps=25010,
    pct_start=0.05,
    cycle_momentum=False,
    anneal_strategy='cos',
    interval='step',
    frequency=1)
load_from = './ckpts/efficientnet-seg-depth.pth'
config_path = '/u/home/caoh/projects/MA_Jiachen/CGFormer/configs/CGFormer-Efficient-Swin-SemanticKITTI.py'
ckpt_path = '/u/home/caoh/projects/MA_Jiachen/CGFormer/ckpt/CGFormer-Efficient-Swin-SemanticKITTI.ckpt'
seed = 7240
log_folder = 'output'
save_path = None
test_mapping = False
submit = False
eval = True
log_every_n_steps = 100
check_val_every_n_epoch = 1
pretrain = False
