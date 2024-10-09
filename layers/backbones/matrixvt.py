# Copyright (c) Megvii Inc. All rights reserved.
import torch
from torch import nn

from mmdet.models.backbones.resnet import BasicBlock

from .base_lss_fpn import BaseLSSFPN

class DepthNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels, depth_channels,
                 camera_aware=False):
        super(DepthNet, self).__init__()
        self.camera_aware = camera_aware

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        if self.camera_aware:
            self.bn = nn.BatchNorm1d(27)
            self.depth_mlp = Mlp(27, mid_channels, mid_channels)
            self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
            self.context_mlp = Mlp(27, mid_channels, mid_channels)
            self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware

        self.context_conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      context_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

    def forward(self, x, mats_dict):
        x = self.reduce_conv(x)

        if self.camera_aware:
            intrins = mats_dict['intrin_mats'][:, 0:1, ..., :3, :3]
            batch_size = intrins.shape[0]
            num_cams = intrins.shape[2]
            ida = mats_dict['ida_mats'][:, 0:1, ...]
            sensor2ego = mats_dict['sensor2ego_mats'][:, 0:1, ..., :3, :]
            bda = mats_dict['bda_mat'].view(batch_size, 1, 1, 4,
                                            4).repeat(1, 1, num_cams, 1, 1)
            mlp_input = torch.cat(
                [
                    torch.stack(
                        [
                            intrins[:, 0:1, ..., 0, 0],
                            intrins[:, 0:1, ..., 1, 1],
                            intrins[:, 0:1, ..., 0, 2],
                            intrins[:, 0:1, ..., 1, 2],
                            ida[:, 0:1, ..., 0, 0],
                            ida[:, 0:1, ..., 0, 1],
                            ida[:, 0:1, ..., 0, 3],
                            ida[:, 0:1, ..., 1, 0],
                            ida[:, 0:1, ..., 1, 1],
                            ida[:, 0:1, ..., 1, 3],
                            bda[:, 0:1, ..., 0, 0],
                            bda[:, 0:1, ..., 0, 1],
                            bda[:, 0:1, ..., 1, 0],
                            bda[:, 0:1, ..., 1, 1],
                            bda[:, 0:1, ..., 2, 2],
                        ],
                        dim=-1,
                    ),
                    sensor2ego.view(batch_size, 1, num_cams, -1),
                ],
                -1,
            )
            mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
            context_se = self.context_mlp(mlp_input)[..., None, None]
            context_img = self.context_se(x, context_se)
            context = self.context_conv(context_img)
            depth_se = self.depth_mlp(mlp_input)[..., None, None]
            depth = self.depth_se(x, depth_se)
            depth = self.depth_conv(depth)
        else:
            context = self.context_conv(x) #(6,80,16,44)
            depth = self.depth_conv(x) #(6,70,16,44)

        return torch.cat([depth, context], dim=1)


class HoriConv(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, cat_dim=0):
        """HoriConv that reduce the image feature
            in height dimension and refine it.

        Args:
            in_channels (int): in_channels
            mid_channels (int): mid_channels
            out_channels (int): output channels
            cat_dim (int, optional): channels of position
                embedding. Defaults to 0.
        """
        super().__init__()

        self.merger = nn.Sequential(
            nn.Conv2d(in_channels + cat_dim,
                      in_channels,
                      kernel_size=1,
                      bias=True),
            nn.Sigmoid(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
        )

        self.reduce_conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv1d(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, pe=None):
        # [N,C,H,W]
        if pe is not None:
            x = self.merger(torch.cat([x, pe], 1))
        else:
            x = self.merger(x)
        x = x.max(2)[0] #(12,80,16,44)->(12,80,44)
        x = self.reduce_conv(x)
        x = self.conv1(x) + x
        x = self.conv2(x) + x
        x = self.out_conv(x)
        return x


class DepthReducer(nn.Module):

    def __init__(self, img_channels, mid_channels):
        """Module that compresses the predicted
            categorical depth in height dimension

        Args:
            img_channels (int): in_channels
            mid_channels (int): mid_channels
        """
        super().__init__()
        self.vertical_weighter = nn.Sequential(
            nn.Conv2d(img_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, feat, depth):
        vert_weight = self.vertical_weighter(feat).softmax(2)  # [N,1,H,W]
        depth = (depth * vert_weight).sum(2)
        return depth


# NOTE Modified Lift-Splat
class MatrixVT(BaseLSSFPN):

    def __init__(
        self,
        x_bound,
        y_bound,
        z_bound,
        d_bound,
        final_dim,
        downsample_factor,
        output_channels,
        img_backbone_conf,
        img_neck_conf,
        depth_net_conf,
        radar_view_transform=False,
        camera_aware=False,
    ):
        """Modified from LSSFPN.

        Args:
            x_bound (list): Boundaries for x.
            y_bound (list): Boundaries for y.
            z_bound (list): Boundaries for z.
            d_bound (list): Boundaries for d.
            final_dim (list): Dimension for input images.
            downsample_factor (int): Downsample factor between feature map
                and input image.
            output_channels (int): Number of channels for the output
                feature map.
            img_backbone_conf (dict): Config for image backbone.
            img_neck_conf (dict): Config for image neck.
            depth_net_conf (dict): Config for depth net.
        """
        super().__init__(
            x_bound,
            y_bound,
            z_bound,
            d_bound,
            final_dim,
            downsample_factor,
            output_channels,
            img_backbone_conf,
            img_neck_conf,
            depth_net_conf,
            camera_aware=False,
            use_da=False,
        )

        self.register_buffer('bev_anchors',
                             self.create_bev_anchors(x_bound, y_bound))
        self.horiconv = HoriConv(self.output_channels, 512,
                                 self.output_channels)
        self.horiconv_pts = HoriConv(self.output_channels, 512,
                                 self.output_channels)
        self.depth_reducer = DepthReducer(self.output_channels,
                                          self.output_channels)
        self.depth_net = self._configure_depth_net(depth_net_conf)
        self.static_mat = None

    def _configure_depth_net(self, depth_net_conf):
        return DepthNet(
            depth_net_conf['in_channels'],
            depth_net_conf['mid_channels'],
            self.output_channels,
            self.depth_channels,
            camera_aware=self.camera_aware
        )
        
    def create_bev_anchors(self, x_bound, y_bound, ds_rate=1):
        """Create anchors in BEV space

        Args:
            x_bound (list): xbound in meters [start, end, step]
            y_bound (list): ybound in meters [start, end, step]
            ds_rate (iint, optional): downsample rate. Defaults to 1.

        Returns:
            anchors: anchors in [W, H, 2]
        """
        x_coords = ((torch.linspace(
            x_bound[0],
            x_bound[1] - x_bound[2] * ds_rate,
            self.voxel_num[0] // ds_rate,
            dtype=torch.float,
        ) + x_bound[2] * ds_rate / 2).view(self.voxel_num[0] // ds_rate,
                                           1).expand(
                                               self.voxel_num[0] // ds_rate,
                                               self.voxel_num[1] // ds_rate))
        y_coords = ((torch.linspace(
            y_bound[0],
            y_bound[1] - y_bound[2] * ds_rate,
            self.voxel_num[1] // ds_rate,
            dtype=torch.float,
        ) + y_bound[2] * ds_rate / 2).view(
            1,
            self.voxel_num[1] // ds_rate).expand(self.voxel_num[0] // ds_rate,
                                                 self.voxel_num[1] // ds_rate))

        anchors = torch.stack([x_coords, y_coords]).permute(1, 2, 0)
        return anchors

    def get_proj_mat(self, mats_dict=None):
        """Create the Ring Matrix and Ray Matrix

        Args:
            mats_dict (dict, optional): dictionary that
                contains intrin- and extrin- parameters.
            Defaults to None.

        Returns:
            tuple: Ring Matrix in [B, D, L, L] and Ray Matrix in [B, W, L, L]
        """
        if self.static_mat is not None:
            return self.static_mat

        bev_size = int(self.voxel_num[0])  # only consider square BEV
        geom_sep = self.get_geometry( # (2,6,70,16,44,3)
            mats_dict['sensor2ego_mats'][:, 0, ...],
            mats_dict['intrin_mats'][:, 0, ...],
            mats_dict['ida_mats'][:, 0, ...],
            mats_dict.get('bda_mat', None),
        )
        geom_sep = (
            geom_sep -
            (self.voxel_coord - self.voxel_size / 2.0)) / self.voxel_size
        geom_sep = geom_sep.mean(3).permute(0, 1, 3, 2,
                                            4).contiguous()  # B,Ncam,W,D,2 (2,6,44,70,3)
        B, Nc, W, D, _ = geom_sep.shape
        geom_sep = geom_sep.long().view(B, Nc * W, D, -1)[..., :2]

        invalid1 = torch.logical_or((geom_sep < 0)[..., 0], (geom_sep < 0)[...,
                                                                           1])
        invalid2 = torch.logical_or((geom_sep > (bev_size - 1))[..., 0],
                                    (geom_sep > (bev_size - 1))[..., 1])
        geom_sep[(invalid1 | invalid2)] = int(bev_size / 2) #(2,264,70,2)
        geom_idx = geom_sep[..., 1] * bev_size + geom_sep[..., 0] #(2,264,70)

        geom_uni = self.bev_anchors[None].repeat([B, 1, 1, 1])  # B,128,128,2
        B, L, L, _ = geom_uni.shape

        circle_map = geom_uni.new_zeros((B, D, L * L))

        ray_map = geom_uni.new_zeros((B, Nc * W, L * L))
        for b in range(B):
            for dir in range(Nc * W):
                ray_map[b, dir, geom_idx[b, dir]] += 1
            for d in range(D):
                circle_map[b, d, geom_idx[b, :, d]] += 1
        null_point = int((bev_size / 2) * (bev_size + 1))
        circle_map[..., null_point] = 0
        ray_map[..., null_point] = 0
        circle_map = circle_map.view(B, D, L * L)
        ray_map = ray_map.view(B, -1, L * L)
        circle_map /= circle_map.max(1)[0].clip(min=1)[:, None]
        ray_map /= ray_map.max(1)[0].clip(min=1)[:, None]

        return circle_map, ray_map #(2,70,16384), (2,264,16384)

    def reduce_and_project(self, feature, depth, mats_dict):
        """reduce the feature and depth in height
            dimension and make BEV feature

        Args:
            feature (Tensor): image feature in [B, C, H, W]
            depth (Tensor): Depth Prediction in [B, D, H, W]
            mats_dict (dict): dictionary that contains intrin-
                and extrin- parameters

        Returns:
            Tensor: BEV feature in B, C, L, L
        """
        # [N,112,H,W], [N,256,H,W]
        depth = self.depth_reducer(feature, depth) #(12,70,44)

        B = mats_dict['intrin_mats'].shape[0]

        # N, C, H, W = feature.shape
        # feature=feature.reshape(N,C*H,W)
        feature = self.horiconv(feature) #(12,80,44)
        # feature = feature.max(2)[0]
        # [N.112,W], [N,C,W]
        depth = depth.permute(0, 2, 1).reshape(B, -1, self.depth_channels) #(12,70,44)->(12,44,70)->(2,264,70)
        feature = feature.permute(0, 2, 1).reshape(B, -1, self.output_channels) #(12,80,44)->(2,264,80)
        circle_map, ray_map = self.get_proj_mat(mats_dict) #(2,70,16384)编码距离信息, (2,264,16384)编码方向信息

        proj_mat = depth.matmul(circle_map) #(2,264,16384)
        proj_mat = (proj_mat * ray_map).permute(0, 2, 1) #(2,16384,264)
        img_feat_with_depth = proj_mat.matmul(feature) #(2,16384,264) * (2,264,80)->(2,16384,80)
        img_feat_with_depth = img_feat_with_depth.permute(0, 2, 1).reshape(
            B, -1, *self.voxel_num[:2])

        return img_feat_with_depth #(2,80,128,128)

    def _forward_single_sweep(self,
                              sweep_index,
                              sweep_imgs,
                              mats_dict,
                              pts_context=None,
                              pts_occupancy=None,
                              is_return_depth=False):
        if self.times is not None:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t3 = torch.cuda.Event(enable_timing=True)
            t4 = torch.cuda.Event(enable_timing=True)
            t5 = torch.cuda.Event(enable_timing=True)
            t1.record()
            torch.cuda.synchronize()
    
        (
            batch_size,
            num_sweeps,
            num_cams,
            num_channels,
            img_height,
            img_width,
        ) = sweep_imgs.shape
        img_feats = self.get_cam_feats(sweep_imgs) #(2,1,6,512,16,44)
        if self.times is not None:
            t2.record()
            torch.cuda.synchronize()
            self.times['img_backbone'].append(t1.elapsed_time(t2))

        source_features = img_feats[:, 0, ...] #(2,6,512,16,44)
        depth_feature = self.depth_net( #(12,150,16,44)
            source_features.reshape(
                batch_size * num_cams,
                source_features.shape[2],
                source_features.shape[3],
                source_features.shape[4],
            ),
            mats_dict,
        )
        if self.times is not None:
            t3.record()
            torch.cuda.synchronize()
            self.times['img_dep'].append(t2.elapsed_time(t3))

        feature = depth_feature[:, self.depth_channels:(
            self.depth_channels + self.output_channels)].float() #(12,80,16,44)
        depth = depth_feature[:, :self.depth_channels].float().softmax(1) #(12,70,16,44)

        img_feat_with_depth = self.reduce_and_project(
            feature, depth, mats_dict)  # [b*n, c, d, w]
        fusion_feature = torch.cat([img_feat_with_depth, pts_context], dim=1) #(12,160,44)

        if self.times is not None:
            t4.record()
            torch.cuda.synchronize()
            self.times['img_transform'].append(t3.elapsed_time(t4))

        if is_return_depth:
            return fusion_feature.contiguous(), depth
        return fusion_feature.contiguous()

    def forward(self,
                sweep_imgs,
                mats_dict,
                ptss_context=None,
                ptss_occupancy=None,
                times=None,
                epoch=None,
                timestamps=None,
                is_return_depth=False):
        """Forward function.

        Args:
            sweep_imgs(Tensor): Input images with shape of (B, num_sweeps,
                num_cameras, 3, H, W).
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps(Tensor): Timestamp for all images with the shape of(B,
                num_sweeps, num_cameras).

        Return:
            Tensor: bev feature map.
        """
        self.times = times
        if self.times is not None:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t1.record()
            torch.cuda.synchronize()

        batch_size, num_sweeps, num_cams, num_channels, img_height, \
            img_width = sweep_imgs.shape
        key_frame_res = self._forward_single_sweep(
            0,
            sweep_imgs[:, 0:1, ...],
            mats_dict,
            ptss_context[:, 0, ...] if ptss_context is not None else None,
            # ptss_occupancy[:, 0, ...] if ptss_occupancy is not None else None,
            is_return_depth=is_return_depth)
        if self.times is not None:
            t2.record()
            torch.cuda.synchronize()
            self.times['img'].append(t1.elapsed_time(t2))

        if num_sweeps == 1:
            if is_return_depth:
                return key_frame_res[0], key_frame_res[1], self.times
            else:
                return key_frame_res, self.times

        key_frame_feature = key_frame_res[0] if is_return_depth else key_frame_res
        ret_feature_list = [key_frame_feature]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map = self._forward_single_sweep( #(2,80,128,128)
                    sweep_index,
                    sweep_imgs[:, sweep_index:sweep_index + 1, ...],
                    mats_dict,
                    ptss_context[:, 0, ...] if ptss_context is not None else None,
                    is_return_depth=False)
                ret_feature_list.append(feature_map)

        if is_return_depth:
            return torch.stack(ret_feature_list, 1), key_frame_res[1], self.times
        else:
            return torch.stack(ret_feature_list, 1), self.times
        

if __name__ == '__main__':
    backbone_conf = {
        'x_bound': [-51.2, 51.2, 0.8],  # BEV grids bounds and size (m)
        'y_bound': [-51.2, 51.2, 0.8],  # BEV grids bounds and size (m)
        'z_bound': [-5, 3, 8],  # BEV grids bounds and size (m)
        'd_bound': [2.0, 58.0,
                    0.5],  # Categorical Depth bounds and division (m)
        'final_dim': (256, 704),  # img size for model input (pix)
        'output_channels':
        80,  # BEV feature channels
        'downsample_factor':
        16,  # ds factor of the feature to be projected to BEV (e.g. 256x704 -> 16x44)  # noqa
        'img_backbone_conf':
        dict(
            type='ResNet',
            depth=50,
            frozen_stages=0,
            out_indices=[0, 1, 2, 3],
            norm_eval=False,
            init_cfg=dict(type='Pretrained',
                          checkpoint='torchvision://resnet50'),
        ),
        'img_neck_conf':
        dict(
            type='SECONDFPN',
            in_channels=[256, 512, 1024, 2048],
            upsample_strides=[0.25, 0.5, 1, 2],
            out_channels=[128, 128, 128, 128],
        ),
        'depth_net_conf':
        dict(in_channels=512, mid_channels=512),
    }

    model = MatrixVT(**backbone_conf)
    # for inference and deployment where intrin & extrin mats are static
    # model.static_mat = model.get_proj_mat(mats_dict)

    bev_feature, depth = model(
        torch.rand((2, 1, 6, 3, 256, 704)), {
            'sensor2ego_mats': torch.rand((2, 1, 6, 4, 4)),
            'intrin_mats': torch.rand((2, 1, 6, 4, 4)),
            'ida_mats': torch.rand((2, 1, 6, 4, 4)),
            'sensor2sensor_mats': torch.rand((2, 1, 6, 4, 4)),
            'bda_mat': torch.rand((2, 4, 4)),
        },
        is_return_depth=True)

    print(bev_feature.shape, depth.shape)
