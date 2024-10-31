import cv2
import numpy as np
import torch
import torch.nn as nn
from utils.basic import matrix_inverse
from torch.cuda.amp.autocast_mode import autocast

from mmdet.models.backbones.resnet import BasicBlock
from .base_lss_fpn import BaseLSSFPN, Mlp, SELayer

from ops.average_voxel_pooling_v2 import average_voxel_pooling

__all__ = ['RVTLSSFPN']


class ViewAggregation(nn.Module):
    """
    Aggregate frustum view features transformed by depth distribution / radar occupancy
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        super(ViewAggregation, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
        )

    @autocast(False)
    def forward(self, x):
        x = self.reduce_conv(x)
        x = self.conv(x)
        x = self.out_conv(x)
        return x


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


class RVTLSSFPN(BaseLSSFPN):
    def __init__(self, export_onnx=False, **kwargs):
        super(RVTLSSFPN, self).__init__(**kwargs)

        self.register_buffer('frustum', self.create_frustum())
        self.z_bound = kwargs['z_bound']
        self.fisheye = kwargs.get('fisheye',  False)
        self.radar_view_transform = kwargs['radar_view_transform']
        self.camera_aware = kwargs['camera_aware']

        self.depth_net = self._configure_depth_net(kwargs['depth_net_conf'])
        self.view_aggregation_net = ViewAggregation(self.output_channels*2,
                                                    self.output_channels*2,
                                                    self.output_channels)
        self.export_onnx = export_onnx

    def _configure_depth_net(self, depth_net_conf):
        return DepthNet(
            depth_net_conf['in_channels'],
            depth_net_conf['mid_channels'],
            self.output_channels,
            self.depth_channels,
            camera_aware=self.camera_aware
        )

    # def get_geometry_collapsed_fisheye(self, sensor2ego_mat, intrin_mat, distort_mats, ida_mat, bda_mat, z_min=-5., z_max=3.):
    #     batch_size, num_cams, _, _ = sensor2ego_mat.shape
    #     points = self.frustum  # (D, H, W, 3)
    #     ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
    #     points = ida_mat.inverse().matmul(points.unsqueeze(-1)).squeeze(-1)
        
    #     _, _, z_dim, x_dim, y_dim, _ = points.shape
    #     # Extract intrinsics and distortion coefficients
    #     cx, cy = intrin_mat[:, :, 0, 2], intrin_mat[:, :, 1, 2]
    #     fx, fy = intrin_mat[:, :, 0, 0], intrin_mat[:, :, 1, 1]
    #     k1, k2, k3, k4 = distort_mats.unsqueeze(-1).unbind(dim=2)
    #     k1, k2, k3, k4 = k1.unsqueeze(2), k2.unsqueeze(2), k3.unsqueeze(2), k4.unsqueeze(2)

    #     # Convert pixel coordinates to normalized coordinates
    #     fx = fx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    #     fy = fy.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    #     cx = cx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    #     cy = cy.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    #     x_normalized = points[:, :, 0, :, :, 0]
    #     y_normalized = points[:, :, 0, :, :, 1]

    #     # Apply Kannala-Brandt projection model
    #     r = torch.sqrt(x_normalized**2 + y_normalized**2) #(2,5,70,32,48)
    #     theta = torch.atan(r)
    #     theta_d = theta * (1 + k1*theta**2 + k2*theta**4 + k3*theta**6 + k4*theta**8)
    #     scale = torch.where(r == 0, torch.tensor(1.0, dtype=torch.float32, device=points.device), r/theta_d)
    #     x_undistorted = x_normalized * scale
    #     y_undistorted = y_normalized * scale

    #     # Compute 3D coordinates
    #     x_3d = x_undistorted.unsqueeze(2).expand(batch_size, num_cams, z_dim, x_dim, y_dim)
    #     y_3d = y_undistorted.unsqueeze(2).expand(batch_size, num_cams, z_dim, x_dim, y_dim)
    #     z_3d = points[..., 2]

    #     # Add padding for homogeneous coordinates
    #     points = torch.stack((x_3d, y_3d, z_3d, torch.ones_like(x_3d)), -1) #(2,5,70,32,48,4)
        
    #     # Adjust coordinates format
    #     points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], 
    #                         points[:, :, :, :, :, 2:]), -1).double().unsqueeze(-1)
        
    #     # From image space to ego space
    #     combine = sensor2ego_mat.matmul(matrix_inverse(intrin_mat)).double()
    #     points = combine.view(batch_size, num_cams, 1, 1, 1, 4, 4).matmul(points).half()

    #     # Apply body transformation if provided
    #     if bda_mat is not None:
    #         bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
    #             batch_size, num_cams, 1, 1, 1, 4, 4)
    #         points = (bda_mat @ points).squeeze(-1)
    #     else:
    #         points = points.squeeze(-1)

    #     # Select valid points based on z-axis range
    #     points_out = points[:, :, :, 0:1, :, :3]
    #     points_valid_z = ((points[..., 2] > z_min) & (points[..., 2] < z_max))

    #     return points_out, points_valid_z
    
    def get_geometry_collapsed_fisheye(self, sensor2ego_mat, intrin_mat, distort_mats, ida_mat, bda_mat, z_min=-5., z_max=3.):
        batch_size, num_cams, _, _ = sensor2ego_mat.shape
        points = self.frustum  # (D, H, W, 3)
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        points = matrix_inverse(ida_mat).matmul(points.unsqueeze(-1)).squeeze(-1).to(torch.float32)
        _, _, z_dim, y_dim, x_dim, _ = points.shape
        
        undistorted_points_all = torch.zeros((batch_size, num_cams, 1, y_dim, x_dim, 2), device=points.device, dtype=points.dtype)
        for b in range(batch_size):
            for cam_idx in range(num_cams):
                K = (intrin_mat[b,cam_idx,:3, :3]).cpu().numpy()
                undistorted_points = cv2.fisheye.undistortPoints\
                    (points[b, cam_idx, 0, :, :, :2].cpu().numpy(), K, \
                        distort_mats[b,cam_idx,:].cpu().numpy(), P=K)
                undistorted_points = torch.tensor(undistorted_points, device=points.device, dtype=points.dtype)
                undistorted_points_all[b, cam_idx, :, :, :, :2] = undistorted_points
        
        undistorted_points_all = undistorted_points_all.repeat(1,1,z_dim,1,1,1)
        z_3d = points[..., 2].unsqueeze(-1)
        
        # Add padding for homogeneous coordinates
        points = torch.cat((undistorted_points_all, z_3d, torch.ones_like(z_3d)), -1).unsqueeze(-1) #(2,5,70,32,48,4,1)
        
        # Adjust coordinates format
        points = torch.cat((points[:, :, :, :, :, :2, :] * points[:, :, :, :, :, 2:3, :], 
                            points[:, :, :, :, :, 2:, :]), 5).double()
        
        # From image space to ego space
        combine = sensor2ego_mat.matmul(matrix_inverse(intrin_mat)).double()
        points = combine.view(batch_size, num_cams, 1, 1, 1, 4, 4).matmul(points).half()

        # Apply body transformation if provided
        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
                batch_size, num_cams, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)

        # Select valid points based on z-axis range
        points_out = points[:, :, :, 0:1, :, :3]
        points_valid_z = ((points[..., 2] > z_min) & (points[..., 2] < z_max))

        return points_out, points_valid_z
    
    def get_geometry_collapsed_raw(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat,
                               z_min=-5., z_max=3.):
        batch_size, num_cams, _, _ = sensor2ego_mat.shape #(1,6,4,4)

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum #(70,32,48,4)
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4) #(2,6,4,4)->(2,6,1,1,1,4,4)
        points = matrix_inverse(ida_mat).matmul(points.unsqueeze(4)).double() #(2,6,1,1,1,4,4)*(70,32,48,4,1)->(2,6,70,32,48,4,1)
        # cam_to_ego
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
             points[:, :, :, :, :, 2:]), 5)

        combine = sensor2ego_mat.matmul(matrix_inverse(intrin_mat)).double() #(2,6,4,4)
        points = combine.view(batch_size, num_cams, 1, 1, 1, 4,
                              4).matmul(points).half() #(2,6,1,1,1,4,4)*(2,6,70,32,48,4,1)->(2,6,70,32,48,4,1)
        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
                batch_size, num_cams, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)

        points_out = points[:, :, :, 0:1, :, :3] #(2,6,70,32,48,4)->(2,6,70,1,48,3)
        points_valid_z = ((points[..., 2] > z_min) & (points[..., 2] < z_max)).to(torch.int32)  #(2,6,70,16,44)

        return points_out, points_valid_z

    def get_geometry_collapsed(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat,
                               z_min=-5., z_max=3.):
        batch_size, num_cams, _, _ = sensor2ego_mat.shape #(1,6,4,4)

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum #(70,32,48,4)
        ##方式一：
        # ida_mat = ida_mat.reshape(batch_size, num_cams, 1, 1, 1, 4, 4) #(2,6,4,4)->(2,6,1,1,1,4,4)
        # points = matrix_inverse(ida_mat).matmul(points.unsqueeze(4)).double() #(2,6,1,1,1,4,4)*(70,32,48,4,1)->(2,6,70,32,48,4,1)
        ## 方式2：
        # ida_mat = ida_mat.reshape(batch_size*num_cams, 1, 4, 4)
        # points = points.unsqueeze(4).reshape(-1, 4,1)
        # points = matrix_inverse(ida_mat).matmul(points)
        # points = points.reshape(1,5,70,32,48,4,1).double()
        ## 方式3
        mul_bs = 10  # 每次处理的深度层数（例如 10 个深度层）
        # 1. 计算 ida_mat 的逆矩阵
        ida_mat_inv = matrix_inverse(ida_mat)
        # 2. 结果容器
        points_transformed_batches = []
        # 3. 按照 mul_bs 逐批处理 points
        for i in range(0, points.shape[0], mul_bs):
            # 获取当前批次的 points，形状为 [mul_bs, 32, 48, 4]
            points_batch = points[i:i + mul_bs]
            # 使用 ida_mat_inv 对 points_batch 进行变换
            points_transformed_batch = torch.matmul(
                ida_mat_inv.reshape(batch_size*num_cams, 1, 4, 4),
                points_batch.unsqueeze(4).reshape(-1, 4, 1)
            )
            points_transformed_batch  = torch.cat((points_transformed_batch[:,:,:2] * points_transformed_batch[:,:,2:3],\
                points_transformed_batch[:,:,2:]), 2) 
            # 恢复原始形状 [1, 5, mul_bs, 32, 48, 4, 1]
            points_transformed_batch = points_transformed_batch.view(batch_size, 5, mul_bs, 32, 48, 4, 1)
            # 将每个批次的结果添加到列表中
            points_transformed_batches.append(points_transformed_batch)
        # 4. 将所有批次的结果拼接起来
        points = torch.cat(points_transformed_batches, dim=2).double()  # 在深度维度上拼接
        
        # # cam_to_ego
        # points = torch.cat(
        #     (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
        #      points[:, :, :, :, :, 2:]), 5)

        combine = sensor2ego_mat.matmul(matrix_inverse(intrin_mat)).double() #(2,6,4,4)
        c1 = combine.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(1,5,70,32,48,4,4)
        points = c1.reshape(-1, 4, 4).matmul(points.reshape(-1, 4, 1))
        points = points.view(-1,5,70,32,48,4,1)
        # combine = combine.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        # points = combine.matmul(points).to(torch.float32) #(2,6,1,1,1,4,4)*(2,6,70,32,48,4,1)->(2,6,70,32,48,4,1)
        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
                batch_size, num_cams, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)

        points_out = points[:, :, :, 0:1, :, :3] #(2,6,70,32,48,4)->(2,6,70,1,48,3)
        points_z = points[..., 2]
        points_valid_z = ((points_z > z_min) & (points_z < z_max)).to(torch.int32) #(2,6,70,16,44)

        return points_out, points_valid_z

    def _forward_view_aggregation_net(self, img_feat_with_depth):
        # BEVConv2D [n, c, d, h, w] -> [n, h, c, w, d]
        img_feat_with_depth = img_feat_with_depth.permute(
            0, 3, 1, 4, 2).contiguous()  # [n, c, d, h, w] -> [n, h, c, w, d]
        n, h, c, w, d = img_feat_with_depth.shape
        img_feat_with_depth = img_feat_with_depth.view(-1, c, w, d)
        img_feat_with_depth = (
            self.view_aggregation_net(img_feat_with_depth).view(
                n, h, c//2, w, d).permute(0, 2, 4, 1, 3).contiguous().float())
        return img_feat_with_depth

    def _forward_depth_net(self, feat, mats_dict):
        return self.depth_net(feat, mats_dict)

    def _split_batch_cam(self, feat, inv=False, num_cams=6):
        batch_size = feat.shape[0]
        if not inv:
            return feat.reshape(batch_size // num_cams, num_cams, *feat.shape[1:])
        else:
            return feat.reshape(batch_size * num_cams, *feat.shape[2:])

    def _forward_single_sweep(self,
                              sweep_index,
                              sweep_imgs,
                              mats,
                              pts_context,
                              pts_occupancy,
                              return_depth=False):
        """Forward function for single sweep.

        Args:
            sweep_index (int): Index of sweeps.
            sweep_imgs (Tensor): Input images.
            mats (list):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego.
                intrin_mats(Tensor): Intrinsic matrix.
                ida_mats(Tensor): Transformation matrix for ida.
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera.
                bda_mat(Tensor): Rotation matrix for bda.
            ptss_context(Tensor): Input point context feature.
            ptss_occupancy(Tensor): Input point occupancy.
            return_depth (bool, optional): Whether to return depth.
                Default: False.

        Returns:
            Tensor: BEV feature map.
        """
        if self.times is not None:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t3 = torch.cuda.Event(enable_timing=True)
            t4 = torch.cuda.Event(enable_timing=True)
            t5 = torch.cuda.Event(enable_timing=True)
            t1.record()
            torch.cuda.synchronize()

        batch_size, num_sweeps, num_cams, num_channels, img_height, \
            img_width = sweep_imgs.shape

        # extract image feature
        img_feats = self.get_cam_feats(sweep_imgs) #(2,1,6,512,16,44)
        if self.times is not None:
            t2.record()
            torch.cuda.synchronize()
            self.times['img_backbone'].append(t1.elapsed_time(t2))

        source_features = img_feats[:, 0, ...] #(1,6,512,16,44)
        source_features = self._split_batch_cam(source_features, inv=True, num_cams=num_cams) #(6,512,16,44)

        # predict image context feature, depth distribution
        depth_feature = self._forward_depth_net( #(6,70+80,16,44)
            source_features,
            mats,
        )
        if self.times is not None:
            t3.record()
            torch.cuda.synchronize()
            self.times['img_dep'].append(t2.elapsed_time(t3))

        image_feature = depth_feature[:, self.depth_channels:(self.depth_channels + self.output_channels)] #(12,80,16,44)

        depth_occupancy = depth_feature[:, :self.depth_channels].softmax( #(12,70,16,44)
            dim=1, dtype=depth_feature.dtype)
        img_feat_with_depth = depth_occupancy.unsqueeze(1) * image_feature.unsqueeze(2) #(12,80,70,16,44)

        # calculate frustum grid within valid height
        if self.fisheye:
            geom_xyz, geom_xyz_valid = self.get_geometry_collapsed_fisheye( #(2,6,70,1,44,3),（2,6,70,16,44）
                mats[0][:, sweep_index, ...], # sensor2ego_mats
                mats[1][:, sweep_index, ...], # intrin_mats
                mats[2][:, sweep_index, ...], # distort_mats
                mats[3][:, sweep_index, ...], # ida_mats
                mats[-1] if len(mats) > 5 else None
                )
        else:
            geom_xyz, geom_xyz_valid = self.get_geometry_collapsed( #(2,6,70,1,44,3),（2,6,70,16,44）
                mats[0][:, sweep_index, ...],
                mats[1][:, sweep_index, ...],
                mats[3][:, sweep_index, ...],
                mats[-1] if len(mats) > 5 else None
            )

        geom_xyz_valid = self._split_batch_cam(geom_xyz_valid, inv=True, num_cams=num_cams).unsqueeze(1)#（12,1,70,16,44）
        img_feat_with_depth = (img_feat_with_depth * geom_xyz_valid).sum(3).unsqueeze(3) #(12,80,70,16,44)->(12,80,70,1,44),可以将无效的体素特征置为零,y方向做了加和

        if self.radar_view_transform:
            radar_occupancy = pts_occupancy.permute(0, 2, 1, 3).contiguous() #(6,70,1,44)
            image_feature_collapsed = (image_feature * geom_xyz_valid.max(2).values).sum(2).unsqueeze(2) #(12,80,1,44)
            img_feat_with_radar = radar_occupancy.unsqueeze(1) * image_feature_collapsed.unsqueeze(2)

            img_context = torch.cat([img_feat_with_depth, img_feat_with_radar], dim=1)
            img_context = self._forward_view_aggregation_net(img_context) #(12,80,70,1,44)
        else:
            img_context = img_feat_with_depth
        if self.times is not None:
            t4.record()
            torch.cuda.synchronize()
            self.times['img_transform'].append(t3.elapsed_time(t4))

        img_context = self._split_batch_cam(img_context, num_cams=num_cams) #(12,80,70,1,44)
        img_context = img_context.permute(0, 1, 3, 4, 5, 2).contiguous() #(2,6,70,1,44,80)

        pts_context = self._split_batch_cam(pts_context, num_cams=num_cams) #(10,80,70,44)->(2,5,80,70,48)
        pts_context = pts_context.unsqueeze(-2).permute(0, 1, 3, 4, 5, 2).contiguous() #(2,6,70,1,44,80)

        fused_context = torch.cat([img_context, pts_context], dim=-1) #（2,5,70,1,44,160）

        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) /
                    self.voxel_size).int()
        geom_xyz[..., 2] = 0  # collapse z-axis, (2,5,70,1,48,3)
        if self.export_onnx:
            return geom_xyz, fused_context.contiguous()
        
        # sparse voxel pooling, 将点或特征映射到离散的体素网格中，并在每个体素内计算特征的平均值
        geo_pos = torch.ones_like(geom_xyz)
        feature_map, _ = average_voxel_pooling(geom_xyz, fused_context.contiguous(), geo_pos,
                                               self.voxel_num.cuda())
        if self.times is not None:
            t5.record()
            torch.cuda.synchronize()
            self.times['img_pool'].append(t4.elapsed_time(t5))

        if return_depth:
            return feature_map.contiguous(), depth_feature[:, :self.depth_channels].softmax(1)
        return feature_map.contiguous() # (2,160,128,128)

    def forward(self,
                sweep_imgs,
                mats,
                ptss_context,
                ptss_occupancy,
                times=None,
                return_depth=False):
        """Forward function.

        Args:
            sweep_imgs(Tensor): Input images with shape of (B, num_sweeps,
                num_cameras, 3, H, W).
            mats(list):
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
            ptss_context(Tensor): Input point context feature with shape of
                (B * num_cameras, num_sweeps, C, D, W).
            ptss_occupancy(Tensor): Input point occupancy with shape of
                (B * num_cameras, num_sweeps, 1, D, W).
            times(Dict, optional): Inference time measurement.
            is_return_depth (bool, optional): Whether to return depth.
                Default: False.

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
            img_width = sweep_imgs.shape #(1,4,6,3,256,704)
        key_frame_res = self._forward_single_sweep(
            0,
            sweep_imgs[:, 0:1, ...],
            mats,
            ptss_context[:, 0, ...] if ptss_context is not None else None,
            ptss_occupancy[:, 0, ...] if ptss_occupancy is not None else None,
            return_depth=return_depth)
        if self.times is not None:
            t2.record()
            torch.cuda.synchronize()
            self.times['img'].append(t1.elapsed_time(t2))

        if num_sweeps == 1:
            if self.export_onnx:
                return key_frame_res[0], key_frame_res[1]
            if return_depth:
                return key_frame_res[0].unsqueeze(1), key_frame_res[1], self.times
            else:
                return key_frame_res.unsqueeze(1), self.times

        if self.export_onnx:
            geo_xyz_ls = [key_frame_res[0]]
            key_frame_feature = key_frame_res[1]
        elif return_depth:
            key_frame_feature = key_frame_res[0]
        else:
            key_frame_feature = key_frame_res
        ret_feature_list = [key_frame_feature]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map = self._forward_single_sweep( #4x(2,160,128,128)
                    sweep_index,
                    sweep_imgs[:, sweep_index:sweep_index + 1, ...],
                    mats,
                    ptss_context[:, sweep_index, ...] if ptss_context is not None else None,
                    ptss_occupancy[:, sweep_index, ...] if ptss_occupancy is not None else None,
                    return_depth=False)
                if self.export_onnx:
                    geo_xyz_ls.append(feature_map[0])
                    feature_map = feature_map[1]
                ret_feature_list.append(feature_map)

        if self.export_onnx:
            return torch.stack(geo_xyz_ls, 1), torch.stack(ret_feature_list, 1)
        if return_depth:
            return torch.stack(ret_feature_list, 1), key_frame_res[1], self.times
        else:
            return torch.stack(ret_feature_list, 1), self.times #(2,4,160,128,128)
