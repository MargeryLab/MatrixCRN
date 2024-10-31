import os
import sys
import onnx
sys.path.append("/maggie.meng/code/CRN")
import torch
import torch.nn as nn

from onnxsim import simplify
from torch.onnx import OperatorExportTypes
from torch.onnx.symbolic_helper import parse_args
from models.camera_radar_net_det import CameraRadarNetDet
from layers.modules.multimodal_deformable_cross_attention import DeformableCrossAttention
from argparse import ArgumentParser

from exps.base_exp import BEVDepthLightningModel
from ops.average_voxel_pooling_v2 import average_voxel_pooling


MAX = None
NUM_SWEEPS = 1

class no_jit_trace:
    def __enter__(self):
        # pylint: disable=protected-access
        self.state = torch._C._get_tracing_state()
        torch._C._set_tracing_state(None)

    def __exit__(self, *args):
        torch._C._set_tracing_state(self.state)
        self.state = None
        

class PillarsScatter(torch.autograd.Function):
    @staticmethod
    @parse_args('v','v','v')
    def symbolic(g, feats, coords, N):
        return g.op('custom_domain::PillarsScatter',feats, coords, N)

    @staticmethod
    def forward(ctx, feats, coords, N):
        return PillarsScatter.pts_middle_encoder(feats[:N], coords[:N], 5)
    
    @staticmethod
    def backward(ctx, grad_output):
        pass   
torch.onnx.register_custom_op_symbolic("custom_domain::PillarsScatter", PillarsScatter.symbolic, 9)


def inverse_symbolic(g, input):
    return g.op("com.microsoft::Inverse", input)
torch.onnx.register_custom_op_symbolic("::inverse", inverse_symbolic, 9)


def nonzero_symbolic(g, input):
    return g.op("com.microsoft::NonZeroPlugin", input)
torch.onnx.register_custom_op_symbolic("::nonzero", nonzero_symbolic, 9)


class SubclassCenterHeadBBox(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    @staticmethod
    def get_bboxes(self, preds_dicts):
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            batch_heatmap = preds_dict["heatmap"].sigmoid()

            batch_reg = preds_dict["reg"]
            batch_hei = preds_dict["height"]

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict["dim"])
            else:
                batch_dim = preds_dict["dim"]

            if "vel" in preds_dict:
                batch_vel = preds_dict["vel"]
            else:
                batch_vel = None
            # (1,1,128,128),(1,2,...),(1,1,...),(1,3,..),(1,2,...), (1,2,128,128)
            rets.extend([batch_heatmap, preds_dict["rot"], batch_hei, batch_dim, batch_vel, batch_reg])
        return rets

    def forward(self, x):
        base_head = self.parent.head
        trunk_outs = [x]
        if base_head.trunk.deep_stem:
            x = base_head.trunk.stem(x)
        else:
            x = base_head.trunk.conv1(x)
            x = base_head.trunk.norm1(x)
            x = base_head.trunk.relu(x)
        for i, layer_name in enumerate(base_head.trunk.res_layers):
            res_layer = getattr(base_head.trunk, layer_name)
            x = res_layer(x)
            if i in base_head.trunk.out_indices:
                trunk_outs.append(x)
        fpn_output = base_head.neck(trunk_outs)
        assert len(fpn_output) == 1    
        head_feat = fpn_output[0]
        ret_dicts = []
        head_feat = base_head.shared_conv(head_feat)
        for task in base_head.task_heads:
            ret_dicts.append(task(head_feat))
        return SubclassCenterHeadBBox.get_bboxes(base_head, ret_dicts)
                        
class TRTModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.head = SubclassCenterHeadBBox(model).cuda()
        
    def forward(self,
                sweep_imgs,
                sensor2ego_mats, intrin_mats, ida_mats,
                radar_voxels,
                radar_num_points,
                radar_coors,
                is_train=False
                ):
        bs, num_sweeps, num_cams, _,_, _ = sweep_imgs.shape
        backbone_pts = self.model.backbone_pts
        pfn_layers = backbone_pts.pts_voxel_encoder.pfn_layers
        ret_context = []
        ret_occupancy = []
        # PillarsScatter.pts_middle_encoder = backbone_pts.pts_middle_encoder
        for sweep_index in range(num_sweeps):
            feats = pfn_layers[0](radar_voxels[sweep_index], export=True) #(5000,8,32)
            feats = pfn_layers[1](feats, export=True) #(5000,8,64)
            N = radar_num_points[sweep_index]
            # radar_bev = PillarsScatter.apply(feats[:N], radar_coors[sweep_index][:N], 5)
            radar_bev = backbone_pts.pts_middle_encoder(feats[:N], radar_coors[sweep_index][:N], 5)
            x = backbone_pts.pts_backbone(radar_bev) #SECOND,(6,64,140,88),(6,128,70,44),(6,64,35,22)
            if backbone_pts.pts_neck is not None:
                x = backbone_pts.pts_neck(x) #SECONDFPN, (6,384,70,44)
            if backbone_pts.return_context:
                x_context = backbone_pts.pred_context(x[-1]).unsqueeze(1) #(6,1,80,70,44)
            if backbone_pts.return_occupancy:
                x_occupancy = backbone_pts.pred_occupancy(x[-1]).unsqueeze(1).sigmoid() #(6,,1,1,70,44)
            ret_context.append(x_context)
            ret_occupancy.append(x_occupancy)
        
        ret_context = torch.cat(ret_context, 1)
        ret_occupancy = torch.cat(ret_occupancy, 1)
        
        backbone_img = self.model.backbone_img
        feats = []
        for sweep_index in range(num_sweeps):
            img_feats = backbone_img.get_cam_feats(sweep_imgs[:, sweep_index:sweep_index + 1, ...]) 
            source_features = img_feats[:, 0, ...] #(1,6,512,16,44)
            source_features = backbone_img._split_batch_cam(source_features, inv=True, num_cams=num_cams) #(6,512,16,44)

            depth_net = backbone_img.depth_net
            x = depth_net.reduce_conv(source_features)
            context = depth_net.context_conv(x)
            depth = depth_net.depth_conv(x)
            depth_feature = torch.cat([depth, context], dim=1)
            image_feature = depth_feature[:, backbone_img.depth_channels:(backbone_img.depth_channels + backbone_img.output_channels)] #(12,80,16,44)

            depth_occupancy = depth_feature[:, :backbone_img.depth_channels].softmax( #(12,70,16,44)
                dim=1, dtype=depth_feature.dtype)
            img_feat_with_depth = depth_occupancy.unsqueeze(1) * image_feature.unsqueeze(2) #(12,80,70,16,44)            
            geom_xyz, geom_xyz_valid = backbone_img.get_geometry_collapsed( #(2,6,70,1,44,3),（2,6,70,16,44）
                sensor2ego_mats[:, sweep_index, ...],
                intrin_mats[:, sweep_index, ...],
                ida_mats[:, sweep_index, ...],
                None,
            )
            
            geom_xyz_valid = backbone_img._split_batch_cam(geom_xyz_valid, inv=True, num_cams=num_cams).unsqueeze(1)#（12,1,70,16,44）
            img_feat_with_depth = (img_feat_with_depth * geom_xyz_valid).sum(3).unsqueeze(3) #(12,80,70,16,44)->(12,80,70,1,44),可以将无效的体素特征置为零,y方向做了加和
            radar_occupancy = ret_occupancy[:, sweep_index, ...].permute(0, 2, 1, 3).contiguous() #(6,70,1,44)
            image_feature_collapsed = (image_feature * geom_xyz_valid.max(2).values).sum(2).unsqueeze(2) #(12,80,1,44)
            img_feat_with_radar = radar_occupancy.unsqueeze(1) * image_feature_collapsed.unsqueeze(2)

            img_context = torch.cat([img_feat_with_depth, img_feat_with_radar], dim=1)
            img_context = backbone_img._forward_view_aggregation_net(img_context) #(12,80,70,1,44)
            img_context = backbone_img._split_batch_cam(img_context, num_cams=num_cams) #(12,80,70,1,44)
            img_context = img_context.permute(0, 1, 3, 4, 5, 2).contiguous() #(2,6,70,1,44,80)

            pts_context = backbone_img._split_batch_cam(ret_context[:, sweep_index, ...], num_cams=num_cams) #(10,80,70,44)->(2,5,80,70,48)
            pts_context = pts_context.unsqueeze(-2).permute(0, 1, 3, 4, 5, 2).contiguous() #(2,6,70,1,44,80)

            fused_context = torch.cat([img_context, pts_context], dim=-1) #（2,6,70,1,44,160）

            geom_xyz = ((geom_xyz - (backbone_img.voxel_coord - backbone_img.voxel_size / 2.0)) /
                        backbone_img.voxel_size).int()
            geom_xyz[..., 2] = 0  # collapse z-axis
            
            geo_pos = torch.ones_like(geom_xyz)
            feature_map, _ = average_voxel_pooling(geom_xyz, fused_context.contiguous(), geo_pos,
                                                backbone_img.voxel_num.cuda()) # (1,160,128,128)
            feats.append(feature_map.contiguous())

        feats = torch.stack(feats, 1)
        fused = self.model.fuser(feats)
        preds = self.head(fused)
        return preds # length = 10*6


def pad(x):
    t = torch.zeros(MAX, *x.shape[1:], dtype=x.dtype, device=x.device)
    t[:x.size(0)] = x
    return t


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
                               default='/maggie.meng/code/CRN/outputs_zongmu/det/CRN_r50_256x704_128x128_4key/lightning_logs/version_24/checkpoints/epoch=29-step=1110.ckpt', type=str)
    parser = BEVDepthLightningModel.add_model_specific_args(parent_parser)
    args = parser.parse_args()
    
    model_engine = model_class(**vars(args))
    model = model_engine.model
    model = model_class.load_from_checkpoint(args.ckpt_path).model
    model.cuda()
    model.eval()

    data_loader = model_engine.predict_dataloader()
    merget_onnx_path = os.path.dirname(args.ckpt_path) + '/crn_merge.onnx'

    trtModel = TRTModel(model).cuda()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            break
        (sweep_imgs, mats, _, gt_boxes_3d, gt_labels_3d, _, depth_labels, pts_pv) = data
        num_sweeps = NUM_SWEEPS
        sweep_imgs = sweep_imgs[:,:num_sweeps,...]
        global MAX
        MAX = 5000 * num_sweeps
        sensor2ego_mats = mats[0].cuda()[:,:num_sweeps, ...]
        intrin_mats = mats[1].cuda()[:,:num_sweeps, ...]
        ida_mats = mats[3].cuda()[:,:num_sweeps, ...]
        pts = pts_pv[:, 0, ...]
        B, N, P, F = pts.shape
        pts = pts.contiguous().view(B*N, P, F)
        voxels, num_points, coors = model_engine.voxelize(pts) #(1508,8,4), (1508),(1508,4)
        N_points = torch.tensor([voxels.size(0)], dtype=torch.int32).repeat(num_sweeps).cuda()
        # N_points_value = voxels.size(0)  # 获取第一个维度的大小
        # N_points = (torch.ones((num_sweeps,), dtype=torch.int32) * N_points_value).cuda()
        features = model.backbone_pts.pts_voxel_encoder(voxels, num_points, coors, do_pfn=False) #(1508,8,7)
        voxels = pad(features).view(num_sweeps, 5000, 8, 7).cuda()
        coords = pad(coors).view(num_sweeps, 5000, 4).cuda()
            
        pre_inputs = (sweep_imgs.cuda(), sensor2ego_mats, intrin_mats, ida_mats, voxels, N_points, coords)
        
        dynamic_axes = {
            'sweep_imgs': {1: 'n_sweeps'}, 
            'sensor2ego_mats': {1: 'n_sweeps'},
            'intrin_mats': {1: 'n_sweeps'},
            'ida_mats': {1: 'n_sweeps'},
            'voxels': {0: 'n_sweeps'},
            'N_points': {0: 'n_sweeps'},
            'coords': {0: 'n_sweeps'},
        }
        torch.onnx.export(
            trtModel,
            pre_inputs,
            merget_onnx_path,
            input_names=["sweep_imgs", "sensor2ego_mats", "intrin_mats", "ida_mats", "voxels", "N_points", "coords"],
            output_names=[f'output_{j}' for j in range(6 * len(model.head.task_heads))],
            # dynamic_axes=dynamic_axes,
            opset_version=15,
            do_constant_folding=True
        )
        print(f"{merget_onnx_path} has saved.")
            
        
        pre_onnx_model = onnx.load(merget_onnx_path)
        # print(onnx.helper.printable_graph(pre_onnx_model.graph))
        try:
            onnx.checker.check_model(pre_onnx_model)
        except Exception:
            print('ONNX Model Incorrect')
        else:
            print('ONNX Model Correct')
            
        pre_model_simp, pre_check = simplify(pre_onnx_model)
        assert pre_check, "Simplified ONNX model could not be validated"
        onnx.save(pre_model_simp, merget_onnx_path)
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
                                       self.head_conf,
                                       export_onnx=self.export_onnx)

    def forward(self, sweep_imgs, mats, pts_pv, is_train=False):
        return self.model(sweep_imgs, mats, sweep_ptss=pts_pv, is_train=is_train) #(1,4,6,3,256,704), (1,4,6,1536,5)


if __name__ == '__main__':
    run_cli(CRNLightningModel,
            'det/CRN_r50_256x704_128x128_4key')
