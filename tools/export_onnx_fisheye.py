import os

import torch
import onnx
from onnxsim import simplify
from mmcv.runner import load_checkpoint

from models.camera_radar_net_det import CameraRadarNetDet
# Copyright (c) Megvii Inc. All rights reserved.
import os
from argparse import ArgumentParser

from utils.torch_dist import all_gather_object, synchronize

from exps.base_exp import BEVDepthLightningModel
from ptq_bev import quantize_net, fuse_conv_bn


def run_cli(model_class=BEVDepthLightningModel,
            exp_name='base_exp',
            use_ema=False,
            ckpt_path=None):
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('-b', '--batch_size_per_device', type=int, default=1)
    parent_parser.add_argument('--device', default='cuda:0')
    parent_parser.add_argument('--seed',
                               type=int,
                               default=0,
                               help='seed for initializing training.')
    parent_parser.add_argument('--ptq',
                               default=False,
                               help='seed for initializing training.')
    parent_parser.add_argument('--ckpt_path', 
                               default='/maggie.meng/code/CRN/outputs_zongmu/det/CRN_r50_256x704_128x128_4key/lightning_logs/version_20_onnx/checkpoints/epoch=29-step=1110.ckpt', type=str)
    parser = BEVDepthLightningModel.add_model_specific_args(parent_parser)
    args = parser.parse_args()
    
    model_engine = model_class(**vars(args))
    model = model_engine.model
    # ckpt     = torch.load(args.ckpt_path, map_location=args.device)
    # model.load_state_dict(ckpt['state_dict'], strict =True)
    model = model_class.load_from_checkpoint(args.ckpt_path).model
    model.cuda()
    model.eval()
    onnx_path = args.ckpt_path.replace('.ckpt', '.onnx')
    data_loader = model_engine.predict_dataloader()
    for i, data in enumerate(data_loader):
        (sweep_imgs, mats, _, gt_boxes_3d, gt_labels_3d, _, depth_labels, pts_pv) = data
        for i, value in enumerate(mats):
            mats[i] = value.cuda()
        with torch.no_grad():
            torch.onnx.export(
                model,
                (sweep_imgs.cuda(), mats, pts_pv.cuda()),
                onnx_path,
                opset_version=11,
                input_names=[
                    'sweep_imgs', 'mats', 'sweep_ptss'
                ],
                output_names=[f'output_{j}' for j in
                                range(10 * len(model.head.task_heads))])
        break
    print(f"finished, {onnx_path} has saved.")
    
    onnx_model = onnx.load(onnx_path)
    try:
        onnx.checker.check_model(onnx_model)
    except Exception:
        print('ONNX Model Incorrect')
    else:
        print('ONNX Model Correct')
        
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"

    onnx.save(model_simp, onnx_path)
    print("saved simplified onnx.")


class CRNLightningModel(BEVDepthLightningModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.data_root = "/defaultShare/tmpnfs/dataset/zm_radar/nuscenes_fmt_with_labels/24-09-20_00-00-01_000_test"
        self.train_info_paths = '/defaultShare/tmpnfs/dataset/zm_radar/nuscenes_fmt_with_labels/24-09-20_00-00-01_000_test/nuscenes_infos_train.pkl'
        self.val_info_paths = '/defaultShare/tmpnfs/dataset/zm_radar/nuscenes_fmt_with_labels/24-09-20_00-00-01_000_test/nuscenes_infos_train.pkl'
        self.predict_info_paths = '/defaultShare/tmpnfs/dataset/zm_radar/nuscenes_fmt_with_labels/24-09-20_00-00-01_000_test/nuscenes_infos_val.pkl'
        self.export_onnx = True
        self.return_image = True
        self.return_depth = True
        self.return_radar_pv = True
        ################################################
        self.optimizer_config = dict(
            type='AdamW',
            lr=2e-4,
            weight_decay=1e-4)
        ################################################
        self.radar_pts_remain_dim = 4
        self.radar_pts_dim = 6
        self.img_conf=dict(img_mean=[123.675, 116.28, 103.53],
                    img_std=[58.395, 57.12, 57.375],
                    to_rgb=True)
        self.ida_aug_conf = {
            'resize_lim': (0.36, 0.48),
            'final_dim': (512, 768),
            'rot_lim': (0., 0.),
            'H': 1280,
            'W': 1920,
            'rand_flip': True,
            'bot_pct_lim': (0.0, 0.0),
            'cams': [
                'CAM_FRONT', 'CAM_AVM_FRONT', 'CAM_AVM_REAR', 'CAM_AVM_LEFT', 'CAM_AVM_RIGHT'
            ],
            'Ncams': 5,
        }
        self.bda_aug_conf = {
            'rot_ratio': 1.0,
            'rot_lim': (-22.5, 22.5),
            'scale_lim': (0.9, 1.1),
            'flip_dx_ratio': 0.5,
            'flip_dy_ratio': 0.5
        }
        ################################################
        self.backbone_img_conf = {
            'fisheye': False,
            'x_bound': [-51.2, 51.2, 0.8],
            'y_bound': [-51.2, 51.2, 0.8],
            'z_bound': [-5, 3, 8],
            'd_bound': [2.0, 58.0, 0.8], #camera坐标系下depth的最大值
            'final_dim': (512, 768),
            'downsample_factor': 16,
            'img_backbone_conf': dict(
                type='ResNet',
                depth=50,
                frozen_stages=0,
                out_indices=[0, 1, 2, 3],
                norm_eval=False,
                init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
            ),
            'img_neck_conf': dict(
                type='SECONDFPN',
                in_channels=[256, 512, 1024, 2048],
                upsample_strides=[0.25, 0.5, 1, 2],
                out_channels=[128, 128, 128, 128],
            ),
            'depth_net_conf':
                dict(in_channels=512, mid_channels=256),
            'radar_view_transform': True,
            'camera_aware': False,
            'output_channels': 80,
        }
        ################################################
        self.backbone_pts_conf = {
            'pts_voxel_layer': dict(
                max_num_points=8,
                voxel_size=[8, 0.4, 2],
                point_cloud_range=[0, 2.0, 0, 768, 58.0, 2],
                max_voxels=(768, 1024)
            ),
            'pts_voxel_encoder': dict(
                type='PillarFeatureNet',
                in_channels=4,
                feat_channels=[32, 64],
                with_distance=False,
                with_cluster_center=False,
                with_voxel_center=True,
                voxel_size=[8, 0.4, 2],
                point_cloud_range=[0, 2.0, 0, 768, 58.0, 2],
                norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                legacy=True
            ),
            'pts_middle_encoder': dict(
                type='PointPillarsScatter',
                in_channels=64,
                output_shape=(140, 96)
            ),
            'pts_backbone': dict(
                type='SECOND',
                in_channels=64,
                out_channels=[64, 128, 256],
                layer_nums=[3, 5, 5],
                layer_strides=[1, 2, 2],
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                conv_cfg=dict(type='Conv2d', bias=True, padding_mode='reflect')
            ),
            'pts_neck': dict(
                type='SECONDFPN',
                in_channels=[64, 128, 256],
                out_channels=[128, 128, 128],
                upsample_strides=[0.5, 1, 2],
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                upsample_cfg=dict(type='deconv', bias=False),
                use_conv_for_no_stride=True
            ),
            'occupancy_init': 0.01,
            'out_channels_pts': 80,
        }
        ################################################
        self.fuser_conf = {
            'img_dims': 80,
            'pts_dims': 80,
            'embed_dims': 128,
            'num_layers': 6,
            'num_heads': 4,
            'bev_shape': (128, 128),
        }
        ################################################
        self.head_conf = {
            'bev_backbone_conf': dict(
                type='ResNet',
                in_channels=128,
                depth=18,
                num_stages=3,
                strides=(1, 2, 2),
                dilations=(1, 1, 1),
                out_indices=[0, 1, 2],
                norm_eval=False,
                base_channels=160,
            ),
            'bev_neck_conf': dict(
                type='SECONDFPN',
                in_channels=[128, 160, 320, 640],
                upsample_strides=[1, 2, 4, 8],
                out_channels=[64, 64, 64, 64]
            ),
            'tasks': [
                dict(num_class=1, class_names=['CAR']),
                dict(num_class=2, class_names=['VAN', 'TRUCK']),
                dict(num_class=2, class_names=['BUS', 'ULTRA_VEHICLE']),
                dict(num_class=1, class_names=['CYCLIST']),
                dict(num_class=2, class_names=['TRICYCLIST','PEDESTRIAN']),
                dict(num_class=2, class_names=['ANIMAL', 'UNKNOWN_MOVABLE']),
                dict(num_class=1, class_names=['ROAD_FENCE']),
                dict(num_class=2, class_names=['TRAFFIC_CONE', 'WATER_FILED_BARRIER']),
                dict(num_class=2, class_names=['LIFTING_LEVERS', 'PILLAR']),
                dict(num_class=1, class_names=['OTHER_BLOCKS']),
            ],
            'common_heads': dict(
                reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
            'bbox_coder': dict(
                type='CenterPointBBoxCoder',
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_num=500,
                score_threshold=0.01,
                out_size_factor=4,
                voxel_size=[0.2, 0.2, 8],
                pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
                code_size=9,
            ),
            'train_cfg': dict(
                point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
                grid_size=[512, 512, 1],
                voxel_size=[0.2, 0.2, 8],
                out_size_factor=4,
                dense_reg=1,
                gaussian_overlap=0.1,
                max_objs=500,
                min_radius=2,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ),
            'test_cfg': dict(
                post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_per_img=500,
                max_pool_nms=False,
                min_radius=[4, 12, 10, 0.85, 0.175, 1, 10, 0.175, 0.175, 1],
                score_threshold=0.01,
                out_size_factor=4,
                voxel_size=[0.2, 0.2, 8],
                nms_type='circle',
                pre_max_size=1000,
                post_max_size=200,
                nms_thr=0.2,
            ),
            'in_channels': 256,  # Equal to bev_neck output_channels.
            'loss_cls': dict(type='GaussianFocalLoss', reduction='mean'),
            'loss_bbox': dict(type='L1Loss', reduction='mean', loss_weight=0.25),
            'gaussian_overlap': 0.1,
            'min_radius': 2,
        }
        ################################################
        self.key_idxes = [-2, -4, -6]
        self.model = CameraRadarNetDet(self.backbone_img_conf,
                                       self.backbone_pts_conf,
                                       self.fuser_conf,
                                       self.head_conf)

    def forward(self, sweep_imgs, mats, pts_pv, is_train=False):
        return self.model(sweep_imgs, mats, sweep_ptss=pts_pv, is_train=is_train) #(1,4,6,3,256,704), (1,4,6,1536,5)


if __name__ == '__main__':
    run_cli(CRNLightningModel,
            'det/CRN_r50_256x704_128x128_4key')
