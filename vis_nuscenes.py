import os
from argparse import ArgumentParser

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mmcv
import numpy as np
from nuscenes.utils.data_classes import Box, LidarPointCloud
from pyquaternion import Quaternion

CLASSES = [
    'CAR','VAN', 'TRUCK', 'BUS', 'ULTRA_VEHICLE', 'CYCLIST', 'TRICYCLIST', 'PEDESTRIAN', 'ANIMAL', 
    'UNKNOWN_MOVABLE', 'ROAD_FENCE', 'TRAFFICCONE', 'WATER_FILED_BARRIER', 'LIFTING_LEVERS', 'PILLAR', 'OTHER_BLOCKS',
    ]
ROOR_PATH = "/defaultShare/tmpnfs/dataset/zm_radar/nuscenes_fmt_with_labels/24-09-20_00-00-01_000_test"
# Set cameras
# IMG_KEYS = [
#     'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
#     'CAM_BACK', 'CAM_BACK_LEFT'
# ]
# CAMERAS_HEIGHT = [1280, 2160, 1280, 1280, 1280, 1280]
# CAMERAS_WIDTH = [1920, 3840, 1920, 1920, 1920, 1920]

IMG_KEYS = [
    'CAM_FRONT', 'CAM_AVM_FRONT', 'CAM_AVM_REAR', 'CAM_AVM_LEFT', 'CAM_AVM_RIGHT'
]
CAMERAS_HEIGHT = [2160, 1280, 1280, 1280, 1280, 1280]
CAMERAS_WIDTH = [3840, 1920, 1920, 1920, 1920, 1920]


def parse_args():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--idx',
                        type=int,
                        help='Index of the dataset to be visualized.')
    parser.add_argument('--result_path', type=str, default='/maggie.meng/code/CRN/outputs_zongmu/det/CRN_r50_256x704_128x128_4key/results_nusc.json', help='Path of the result json file.')
    # parser.add_argument('--result_path', type=str, default='/maggie.meng/code/CRN/outputs/det/CRN_r50_256x704_128x128_4key/results_nusc.json', help='Path of the result json file.')
    parser.add_argument('--target_path', type=str, default="/maggie.meng/code/CRN/fisheye_vis_1.2k",
                        help='Target path to save the visualization result.')
    # parser.add_argument('--pkl_path', type=str, default='/defaultShare/tmpnfs/dataset/zm_radar/nuscenes_gen/24-09-04_2/nuscenes_infos_test.pkl')
    parser.add_argument('--pkl_path', type=str, default='/defaultShare/tmpnfs/dataset/zm_radar/nuscenes_fmt_with_labels/24-09-20_00-00-01_000_test/nuscenes_infos_train.pkl')

    args = parser.parse_args()
    return args


def get_ego_box(box_dict, ego2global_rotation, ego2global_translation):
    box = Box(
        box_dict['translation'],
        box_dict['size'],
        Quaternion(box_dict['rotation']),
    )
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    box.translate(trans)
    box.rotate(rot)
    box_xyz = np.array(box.center)
    box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
    box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
    box_velo = np.array(box.velocity[:2])
    return np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros(points.shape[0])
    ones = np.ones(points.shape[0])
    rot_matrix = np.stack(
        (cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones),
        axis=1).reshape(-1, 3, 3)
    points_rot = np.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot


def get_corners(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading],
            (x, y, z) is the box center
    Returns:
    """
    template = (np.array((
        [1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
        [-1, 1, -1],
        [1, 1, 1],
        [1, -1, 1],
        [-1, -1, 1],
        [-1, 1, 1],
    )) / 2)

    corners3d = np.tile(boxes3d[:, None, 3:6],
                        [1, 8, 1]) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.reshape(-1, 8, 3),
                                      boxes3d[:, 6]).reshape(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d


def get_bev_lines(corners):
    return [[[corners[i, 0], corners[(i + 1) % 4, 0]],
             [corners[i, 1], corners[(i + 1) % 4, 1]]] for i in range(4)]


def get_3d_lines(corners):
    ret = []
    for st, ed in [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7],
                   [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
        if corners[st, -1] > 0 and corners[ed, -1] > 0:
            ret.append([[corners[st, 0], corners[ed, 0]],
                        [corners[st, 1], corners[ed, 1]]])
    return ret


def undistort_points(points_2d, cam):
    fx, fy = cam['fx'], cam['fy']
    cx, cy = cam['cx'], cam['cy']
    k1, k2, k3, k4 = cam['k1'], cam['k2'], cam['k3'], cam['k4']
    
    # Convert pixel coordinates to normalized coordinates
    x = (points_2d[:, 0] - cx) / fx
    y = (points_2d[:, 1] - cy) / fy
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan(r)
    
    # Apply distortion model
    theta_d = theta * (1 + k1*theta**2 + k2*theta**4 + k3*theta**6 + k4*theta**8)
    
    scale = np.where(r == 0, 1.0, theta_d / r)
    x_undistorted = x * scale
    y_undistorted = y * scale
    
    # Convert back to pixel coordinates
    undistorted_points = np.stack((x_undistorted * fx + cx, y_undistorted * fy + cy), axis=-1)
    
    return undistorted_points


def get_cam_corners(corners, translation, rotation, cam_intrinsics, k, cam_key):
    cam_corners = corners.copy()
    cam_corners -= np.array(translation)
    cam_corners = cam_corners @ Quaternion(rotation).inverse.rotation_matrix.T
    cam_corners = cam_corners @ np.array(cam_intrinsics).T
    valid = cam_corners[:, -1] > 0
    cam_corners /= cam_corners[:, 2:3]
    cam_corners[~valid] = 0
    
    # if 'AVM' in cam_key:
    fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
    cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]
    cam_corners[:, :2] = undistort_points(cam_corners, {
        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
        'k1': k[0], 'k2': k[1], 'k3': k[2], 'k4': k[3]
    })   
    return cam_corners


def demo(
    idx,
    nusc_results_file,
    dump_file,
    pkl_path,
    # root_path="/defaultShare/tmpnfs/dataset/zm_radar/nuscenes_gen/24-09-04_2",
    root_path=ROOR_PATH,
    threshold=0.32,
    show_range=100,
    show_classes=CLASSES,
):
    # Get data from dataset
    results = mmcv.load(nusc_results_file)['results']
    infos = mmcv.load(pkl_path)
    for info in infos:
        lidar_path = info['lidar_infos']['LIDAR_TOP']['filename']
        lidar_points = np.fromfile(os.path.join(root_path, lidar_path),
                                dtype=np.float32,
                                count=-1).reshape(-1, 5)[..., :4]
        lidar_calibrated_sensor = info['lidar_infos']['LIDAR_TOP'][
            'calibrated_sensor']
        # Get point cloud
        pts = lidar_points.copy()
        ego2global_rotation = np.mean(
            [info['cam_infos'][cam]['ego_pose']['rotation'] for cam in IMG_KEYS],
            0)
        ego2global_translation = np.mean([
            info['cam_infos'][cam]['ego_pose']['translation'] for cam in IMG_KEYS
        ], 0)
        lidar_points = LidarPointCloud(lidar_points.T)
        lidar_points.rotate(
            Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
        lidar_points.translate(np.array(lidar_calibrated_sensor['translation']))
        pts = lidar_points.points.T

        # # Get GT corners
        # gt_corners = []
        # for i in range(len(info['ann_infos'])):
        #     if map_name_from_general_to_detection[
        #             info['ann_infos'][i]['category_name']] in show_classes:
        #         box = get_ego_box(
        #             dict(
        #                 size=info['ann_infos'][i]['size'],
        #                 rotation=info['ann_infos'][i]['rotation'],
        #                 translation=info['ann_infos'][i]['translation'],
        #             ), ego2global_rotation, ego2global_translation)
        #         if np.linalg.norm(box[:2]) <= show_range:
        #             corners = get_corners(box[None])[0]
        #             gt_corners.append(corners)

        # Get prediction corners
        pred_corners, pred_class = [], []
        for box in results[info['sample_token']]:
            if box['detection_score'] >= threshold and box[
                    'detection_name'] in show_classes:
                box3d = get_ego_box(box, ego2global_rotation,
                                    ego2global_translation)
                box3d[2] += 0.5 * box3d[5]  # NOTE
                if np.linalg.norm(box3d[:2]) <= show_range:
                    corners = get_corners(box3d[None])[0]
                    pred_corners.append(corners)
                    pred_class.append(box['detection_name'])

        # Set figure size
        plt.figure(figsize=(24, 8))

        for i, k in enumerate(IMG_KEYS):
            # Draw camera views
            fig_idx = i + 1 if i < 3 else i + 2
            plt.subplot(2, 4, fig_idx)

            # Set camera attributes
            plt.title(k)
            plt.axis('off')
            plt.xlim(0, CAMERAS_WIDTH[i])
            plt.ylim(CAMERAS_HEIGHT[i], 0)

            img = mmcv.imread(
                os.path.join(root_path, info['cam_infos'][k]['filename']))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Draw images
            plt.imshow(img)

            # Draw 3D predictions
            for corners, cls in zip(pred_corners, pred_class):
                cam_corners = get_cam_corners(
                    corners,
                    info['cam_infos'][k]['calibrated_sensor']['translation'],
                    info['cam_infos'][k]['calibrated_sensor']['rotation'],
                    np.array(info['cam_infos'][k]['calibrated_sensor']['camera_intrinsic']),
                    info['cam_infos'][k]['calibrated_sensor']['camera_distortion'],
                    k)
                lines = get_3d_lines(cam_corners)
                for line in lines:
                    plt.plot(line[0],
                            line[1],
                            c=cm.get_cmap('tab10')(show_classes.index(cls)))

        # Draw BEV
        plt.subplot(1, 4, 4)

        # Set BEV attributes
        plt.title('LIDAR_TOP')
        plt.axis('equal')
        plt.xlim(-40, 40)
        plt.ylim(-40, 40)

        # Draw point cloud
        plt.scatter(-pts[:, 1], pts[:, 0], s=0.01, c=pts[:, -1], cmap='gray')

        # # Draw BEV GT boxes
        # for corners in gt_corners:
        #     lines = get_bev_lines(corners)
        #     for line in lines:
        #         plt.plot([-x for x in line[1]],
        #                 line[0],
        #                 c='r',
        #                 label='ground truth')

        # Draw BEV predictions
        for corners in pred_corners:
            lines = get_bev_lines(corners)
            for line in lines:
                plt.plot([-x for x in line[1]], line[0], c='g', label='prediction')

        # Set legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(),
                by_label.keys(),
                loc='upper right',
                framealpha=1)

        # Save figure
        plt.tight_layout(w_pad=0, h_pad=2)
        filename = os.path.basename(info['cam_infos'][k]['filename'])
        plt.savefig(os.path.join(dump_file, filename))
        print(filename)


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.target_path):
        os.mkdir(args.target_path)
    demo(
        args.idx,
        args.result_path,
        args.target_path,
        args.pkl_path
    )