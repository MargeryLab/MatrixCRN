import torch
import torch.nn.functional as F
from torch import nn

from mmcv.cnn import bias_init_with_prob
from mmdet3d.models import builder
from mmcv.ops import Voxelization
from mmcv.runner import BaseModule, force_fp32


class PtsBackboneCamCoords(nn.Module):
    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PFNLayers.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool, optional): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 pts_voxel_layer,
                 pts_voxel_encoder,
                 pts_middle_encoder,
                 pts_backbone,
                 pts_neck,
                 return_context=True,
                 return_occupancy=True,
                 export_onnx=False,
                 **kwargs,
                 ):
        super(PtsBackboneCamCoords, self).__init__()

        self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        self.pts_voxel_encoder = builder.build_voxel_encoder(pts_voxel_encoder)
        self.pts_middle_encoder = builder.build_middle_encoder(pts_middle_encoder)
        self.pts_backbone = builder.build_backbone(pts_backbone)
        self.return_context = return_context
        self.return_occupancy = return_occupancy
        mid_channels = pts_backbone['out_channels'][-1]
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)
            mid_channels = sum(pts_neck['out_channels'])
        else:
            self.pts_neck = None

        if self.return_context:
            if 'out_channels_pts' in kwargs:
                out_channels = kwargs['out_channels_pts']
            else:
                out_channels = 80
            self.pred_context = nn.Sequential(
                nn.Conv2d(mid_channels,
                          mid_channels//2,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          padding_mode='reflect'),
                nn.BatchNorm2d(mid_channels//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels//2,
                          out_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0),
            )

        if self.return_occupancy:
            self.pred_occupancy = nn.Sequential(
                nn.Conv2d(mid_channels,
                          mid_channels//2,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          padding_mode='reflect'),
                nn.BatchNorm2d(mid_channels//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels//2,
                          1,
                          kernel_size=1,
                          stride=1,
                          padding=0),
            )

            if 'occupancy_init' in kwargs:
                occupancy_init = kwargs['occupancy_init']
            else:
                occupancy_init = 0.01
            self.pred_occupancy[-1].bias.data.fill_(bias_init_with_prob(occupancy_init))
            
            self.export_onnx = export_onnx

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        bs, num_sweeps, num_cam, _, f_dim = points.shape
        vox_bs = bs * num_sweeps * num_cam
        points = points.contiguous().view(vox_bs, -1, f_dim)
        for t in range(vox_bs):
            ret = self.pts_voxel_layer(points[t])
            if len(ret) == 3:
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=t))

            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)

        return feats, coords, sizes
    
    def _forward_single_sweep(self, voxels, num_points, coors, batch_size=6):
        if self.times is not None:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t3 = torch.cuda.Event(enable_timing=True)
            t4 = torch.cuda.Event(enable_timing=True)
            # t1.record()
            t2.record()
            torch.cuda.synchronize()
            
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors) #PillarFeatureNet, (2813,64)
        x = self.pts_middle_encoder(voxel_features, coors, batch_size) #PointPillarsScatter(10,64,140,88)
        x = self.pts_backbone(x) #SECOND,(10,64,140,88),(10,128,70,44),(10,64,35,22)
        if self.pts_neck is not None:
            x = self.pts_neck(x) #SECONDFPN, (10,384,70,44)

        if self.times is not None:
            t3.record()
            torch.cuda.synchronize()
            self.times['pts_backbone'].append(t2.elapsed_time(t3))

        x_context = None
        x_occupancy = None
        if self.return_context:
            x_context = self.pred_context(x[-1]).unsqueeze(1) #(6,1,80,70,44)
        if self.return_occupancy:
            x_occupancy = self.pred_occupancy(x[-1]).unsqueeze(1).sigmoid() #(6,,1,1,70,44)

        if self.times is not None:
            t4.record()
            torch.cuda.synchronize()
            self.times['pts_head'].append(t3.elapsed_time(t4))

        return x_context, x_occupancy
    
    def forward_simple(self, ptss, radar_num_points, radar_coors, batch_size=6, num_sweeps=4, times=None):
        self.times = times
        if self.times is not None:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t3 = torch.cuda.Event(enable_timing=True)
            t4 = torch.cuda.Event(enable_timing=True)
            t1.record()
            torch.cuda.synchronize()        
            
        with torch.no_grad():
            radar_voxels, coords, sizes = self.voxelize(ptss)
            batch_size = coords[-1, 0] + 1
            voxel_features = self.pts_voxel_encoder(radar_voxels, sizes, coords) #PillarFeatureNet
            x = self.pts_middle_encoder(voxel_features, coords, batch_size) #PointPillarsScatter(40,64,140,88)
            x = self.pts_backbone(x) #SECOND,(6,64,140,88),(6,128,70,44),(6,64,35,22)
            if self.pts_neck is not None:
                x = self.pts_neck(x) #SECONDFPN, (6,384,70,44)

            if self.times is not None:
                t2.record()
                torch.cuda.synchronize()
                self.times['pts_backbone'].append(t1.elapsed_time(t2))

            x_context = None
            x_occupancy = None
            if self.return_context:
                x_context = self.pred_context(x[-1]).unsqueeze(1) #(6,1,80,70,44)
            if self.return_occupancy:
                x_occupancy = self.pred_occupancy(x[-1]).unsqueeze(1).sigmoid() #(6,,1,1,70,44)

            if self.times is not None:
                t3.record()
                torch.cuda.synchronize()
                self.times['pts_head'].append(t2.elapsed_time(t3))

        if self.times is not None:
            t4.record()
            torch.cuda.synchronize()
            self.times['pts'].append(t1.elapsed_time(t4))

        ret_context = None
        ret_occupancy = None
        if self.return_context:
            ret_context = x_context
        if self.return_occupancy:
            ret_occupancy = x_occupancy
        if self.export_onnx:
            return ret_context, ret_occupancy
        
        return ret_context, ret_occupancy, self.times #(40,1,80,70,48)

    def forward(self, radar_voxels, radar_num_points, radar_coors, batch_size=6, num_sweeps=4, times=None):
        self.times = times
        if self.times is not None:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t1.record()
            torch.cuda.synchronize()

        key_context, key_occupancy = self._forward_single_sweep\
            (radar_voxels[0], radar_num_points[0], radar_coors[0], batch_size=batch_size) #(6,1,80,70,44),(6,1,1,70,44)
        
        if self.times is not None:
            t2.record()
            torch.cuda.synchronize()
            self.times['pts'].append(t1.elapsed_time(t2))

        if num_sweeps == 1:
            if self.export_onnx:
                return key_context, key_occupancy
            return key_context, key_occupancy, self.times

        context_list = [key_context]
        occupancy_list = [key_occupancy]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                context, occupancy = self._forward_single_sweep\
                    (radar_voxels[sweep_index], radar_num_points[sweep_index], radar_coors[sweep_index], batch_size=batch_size)
                context_list.append(context)
                occupancy_list.append(occupancy)

        ret_context = None
        ret_occupancy = None
        if self.return_context:
            ret_context = torch.cat(context_list, 1)
        if self.return_occupancy:
            ret_occupancy = torch.cat(occupancy_list, 1)
        if self.export_onnx:
            return ret_context, ret_occupancy
        
        return ret_context, ret_occupancy, self.times #(10,4,80,70,48)


class PtsBackbone(nn.Module):
    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PFNLayers.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool, optional): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 pts_voxel_layer,
                 pts_voxel_encoder,
                 pts_middle_encoder,
                 pts_backbone,
                 pts_neck,
                 return_context=True,
                 return_occupancy=False,
                 **kwargs,
                 ):
        super(PtsBackbone, self).__init__()

        self.pts_voxel_encoder = builder.build_voxel_encoder(pts_voxel_encoder)
        self.pts_middle_encoder = builder.build_middle_encoder(pts_middle_encoder)
        self.pts_backbone = builder.build_backbone(pts_backbone)
        self.return_context = return_context
        self.return_occupancy = return_occupancy
        mid_channels = pts_backbone['out_channels'][-1]
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)
            mid_channels = sum(pts_neck['out_channels'])
        else:
            self.pts_neck = None

        if self.return_context:
            if 'out_channels_pts' in kwargs:
                out_channels = kwargs['out_channels_pts']
            else:
                out_channels = 80
            self.pred_context = nn.Sequential(
                nn.Conv2d(mid_channels,
                          mid_channels//2,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          padding_mode='reflect'),
                nn.BatchNorm2d(mid_channels//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels//2,
                          out_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0),
            )

        if self.return_occupancy:
            self.pred_occupancy = nn.Sequential(
                nn.Conv2d(mid_channels,
                          mid_channels//2,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          padding_mode='reflect'),
                nn.BatchNorm2d(mid_channels//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels//2,
                          1,
                          kernel_size=1,
                          stride=1,
                          padding=0),
            )

            if 'occupancy_init' in kwargs:
                occupancy_init = kwargs['occupancy_init']
            else:
                occupancy_init = 0.01
            self.pred_occupancy[-1].bias.data.fill_(bias_init_with_prob(occupancy_init))

    def _forward_single_sweep(self, voxels, num_points, coors, batch_size=6):
        if self.times is not None:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t3 = torch.cuda.Event(enable_timing=True)
            t4 = torch.cuda.Event(enable_timing=True)
            # t1.record()
            t2.record()
            torch.cuda.synchronize()

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors) #PillarFeatureNet, (3188,64)
        x = self.pts_middle_encoder(voxel_features, coors, batch_size) #PointPillarsScatter(2,64,256,256)
        x = self.pts_backbone(x) #SECOND,(6,64,140,88),(6,128,70,44),(6,64,35,22)
        if self.pts_neck is not None:
            x = self.pts_neck(x) #SECONDFPN, (10,384,70,44)

        if self.times is not None:
            t3.record()
            torch.cuda.synchronize()
            self.times['pts_backbone'].append(t2.elapsed_time(t3))

        x_context = None
        x_occupancy = None
        if self.return_context:
            x_context = self.pred_context(x[-1]).unsqueeze(1) #(10,1,80,70,44)
        if self.return_occupancy:
            x_occupancy = self.pred_occupancy(x[-1]).unsqueeze(1).sigmoid() #(6,,1,1,70,44)

        if self.times is not None:
            t4.record()
            torch.cuda.synchronize()
            self.times['pts_head'].append(t3.elapsed_time(t4))

        return x_context, x_occupancy

    def forward(self, radar_voxels, radar_num_points, radar_coors, times=None):
        self.times = times
        if self.times is not None:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t1.record()
            torch.cuda.synchronize()

        key_context, key_occupancy = self._forward_single_sweep\
            (radar_voxels[0], radar_num_points[0], radar_coors[0], batch_size=batch_size) #(6,1,80,70,44),(6,1,1,70,44)
        
        if self.times is not None:
            t2.record()
            torch.cuda.synchronize()
            self.times['pts'].append(t1.elapsed_time(t2))

        if num_sweeps == 1:
            return key_context, key_occupancy, self.times

        context_list = [key_context]
        occupancy_list = [key_occupancy]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                context, occupancy = self._forward_single_sweep\
                    (radar_voxels[sweep_index], radar_num_points[sweep_index], radar_coors[sweep_index], batch_size=batch_size)
                context_list.append(context)
                occupancy_list.append(occupancy)

        ret_context = None
        ret_occupancy = None
        if self.return_context:
            ret_context = torch.cat(context_list, 1)
        if self.return_occupancy:
            ret_occupancy = torch.cat(occupancy_list, 1)
        return ret_context, ret_occupancy, self.times #(2,4,80,128,128)
