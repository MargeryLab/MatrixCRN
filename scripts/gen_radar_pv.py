import math
import os
import cv2
import mmcv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points


DATA_PATH = '/defaultShare/tmpnfs/dataset/zm_radar/nuscenes_fmt_with_labels/24-09-20_00-00-01_000_test'
RADAR_SPLIT = 'radar_bev_filter'
OUT_PATH = 'radar_pv_filter'
info_paths = ['/defaultShare/tmpnfs/dataset/zm_radar/nuscenes_fmt_with_labels/24-09-20_00-00-01_000_test/nuscenes_infos_train.pkl', 
              '/defaultShare/tmpnfs/dataset/zm_radar/nuscenes_fmt_with_labels/24-09-20_00-00-01_000_test/nuscenes_infos_val.pkl']

# DATA_PATH = '/defaultShare/tmpnfs/dataset/zm_radar/nuscenes_gen/24-09-04_2'
# RADAR_SPLIT = 'radar_bev_filter_test'
# OUT_PATH = 'radar_pv_filter'
# info_paths = ['/defaultShare/tmpnfs/dataset/zm_radar/nuscenes_gen/24-09-04_2/nuscenes_infos_test.pkl']

MIN_DISTANCE = 0.1
MAX_DISTANCE = 100.

# IMG_SHAPE = [(900, 1600),(900, 1600),(900, 1600),(900, 1600),(900, 1600),(900, 1600)]
IMG_SHAPES = [(2160, 3840), (1280, 1920), (1280, 1920), (1280, 1920), (1280, 1920),(1280, 1920), 
              (1280, 1920),(1280, 1920),(1280, 1920),(1280, 1920)]

# MAX_DIM = 7
MAX_DIM = 6

lidar_key = 'LIDAR_TOP'
cam_keys = [
            'CAM_FRONT', 
            'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT', 
            'CAM_AVM_FRONT', 'CAM_AVM_REAR', 'CAM_AVM_LEFT', 'CAM_AVM_RIGHT'
]

def plot_on_image(file_path, projected_points, vis_dir="tmp"):
    """将投影点绘制到图像上"""
    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)
    
    assert os.path.exists(file_path)
    file_name = os.path.split(file_path)[-1]
    image = cv2.imread(file_path)
    n_points = projected_points.shape[1]
    if n_points != 0:
        for i in range(n_points):
            x = int(projected_points[0, i])
            y = int(projected_points[1, i])
            # 绘制投影点，确保坐标在图像范围内
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)  # 红色点

        save_path = os.path.join(vis_dir, file_name)
        cv2.imwrite(save_path, image)
        print(f'Projection image saved as {save_path}')
    else:
        print(f'No any points in {file_path}')
        

def undistort_points(points_2d, cam):
    fx, fy = cam['fx'], cam['fy']
    cx, cy = cam['cx'], cam['cy']
    k1, k2, k3, k4 = cam['k1'], cam['k2'], cam['k3'], cam['k4']
    
    # Convert pixel coordinates to normalized coordinates
    x = (points_2d[0, :] - cx) / fx
    y = (points_2d[1, :] - cy) / fy
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan(r)
    
    # Apply distortion model
    theta_d = theta * (1 + k1*theta**2 + k2*theta**4 + k3*theta**6 + k4*theta**8)
    
    scale = np.where(r == 0, 1.0, theta_d / r)
    x_undistorted = x * scale
    y_undistorted = y * scale
    
    # Convert back to pixel coordinates
    undistorted_points = np.vstack((x_undistorted * fx + cx, y_undistorted * fy + cy))
    
    return undistorted_points


def view_points_fisheye(points: np.ndarray, camera_intrinsic: np.ndarray, k: np.ndarray, cam_key, normalize: bool) -> np.ndarray:
    """
    Map 3D points to 2D plane with fisheye distortion correction.
    
    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param camera_intrinsic: <np.float32: 3, 3> Camera intrinsic parameters.
    :param k: <np.float32: 4> Distortion coefficients.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point.
    """
    assert camera_intrinsic.shape[0] <= 4
    assert camera_intrinsic.shape[1] <= 4
    assert points.shape[0] == 3

    fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
    cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]
    
    # Create a 4x4 identity matrix for homogeneous coordinates
    viewpad = np.eye(4)
    viewpad[:camera_intrinsic.shape[0], :camera_intrinsic.shape[1]] = camera_intrinsic

    # Do operation in homogeneous coordinates
    nbr_points = points.shape[1]
    points_homogeneous = np.concatenate((points, np.ones((1, nbr_points))))
    
    # Project points onto the image plane
    projected_points_homogeneous = np.dot(viewpad, points_homogeneous)
    projected_points_homogeneous = projected_points_homogeneous[:3, :]
    
    # Normalize to get image coordinates
    points_2d = projected_points_homogeneous / projected_points_homogeneous[2:3, :].repeat(3, 0).reshape(3, nbr_points)
    # if 'AVM' in cam_key:
    # Apply undistortion
    points_2d[:2, :] = undistort_points(points_2d, {
        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
        'k1': k[0], 'k2': k[1], 'k3': k[2], 'k4': k[3]
    })

    return points_2d


def map_pointcloud_to_fisheye_image(
    pc,
    features,
    img_shape,
    cam_calibrated_sensor,
    cam_ego_pose,
    cam_key
):
    pc = LidarPointCloud(pc)

    # Third step: transform from global into the ego vehicle
    # frame for the timestamp of the image.
    pc.translate(-np.array(cam_ego_pose['translation']))
    pc.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    pc.translate(-np.array(cam_calibrated_sensor['translation']))
    pc.rotate(Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]
    features = np.concatenate((depths[:, None], features), axis=1) #(1795,5)

    # Take the actual picture (matrix multiplication with camera-matrix
    # + renormalization).
    camera_intrinsic = np.array(cam_calibrated_sensor['camera_intrinsic'])
    distortion_coefficient = np.array(cam_calibrated_sensor['camera_distortion'])
    points = view_points_fisheye(pc.points[:3, :],
                         camera_intrinsic,
                         distortion_coefficient,
                         cam_key,
                         normalize=True)

    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons. Also make
    # sure points are at least 1m in front of the camera to avoid
    # seeing the lidar points on the camera casing for non-keyframes
    # which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > MIN_DISTANCE)
    mask = np.logical_and(mask, depths < MAX_DISTANCE)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < img_shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < img_shape[0] - 1)
    points = points[:, mask]
    features = features[mask]

    return points, features # 投影到图像坐标系的雷达点云（3x614，第三维都是1）,features(614x4, depth, vx, vy, sweep_idx)


# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py#L834
def map_pointcloud_to_image(
    pc,
    features,
    img_shape,
    cam_calibrated_sensor,
    cam_ego_pose,
):
    pc = LidarPointCloud(pc)

    # Third step: transform from global into the ego vehicle
    # frame for the timestamp of the image.
    pc.translate(-np.array(cam_ego_pose['translation']))
    pc.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    pc.translate(-np.array(cam_calibrated_sensor['translation']))
    pc.rotate(Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]
    features = np.concatenate((depths[:, None], features), axis=1) #(1795,5)

    # Take the actual picture (matrix multiplication with camera-matrix
    # + renormalization).
    points = view_points(pc.points[:3, :],
                         np.array(cam_calibrated_sensor['camera_intrinsic']),
                         normalize=True)

    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons. Also make
    # sure points are at least 1m in front of the camera to avoid
    # seeing the lidar points on the camera casing for non-keyframes
    # which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > MIN_DISTANCE)
    mask = np.logical_and(mask, depths < MAX_DISTANCE)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < img_shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < img_shape[0] - 1)
    points = points[:, mask]
    features = features[mask]

    return points, features # 投影到图像坐标系的雷达点云（3x614，第三维都是1）， features(614x4, xyz depth)


def worker(info):
    radar_file_name = os.path.split(info['lidar_infos']['LIDAR_TOP']['filename'])[-1]
    points = np.fromfile(os.path.join(DATA_PATH, RADAR_SPLIT, radar_file_name),
                         dtype=np.float32,
                         count=-1).reshape(-1, MAX_DIM)

    lidar_calibrated_sensor = info['lidar_infos'][lidar_key][
        'calibrated_sensor']
    lidar_ego_pose = info['lidar_infos'][lidar_key]['ego_pose']

    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.
    pc = LidarPointCloud(points[:, :4].T)  # use 4 dims for code compatibility
    features = points[:, 3:] #(1795,7)->(1795, 4),rcs, vx_comp, vy_comp, (dummy field for sweep info)

    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor['translation']))

    # Second step: transform from ego to the global frame.
    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation']))

    for i, cam_key in enumerate(cam_keys):
        cam_calibrated_sensor = info['cam_infos'][cam_key]['calibrated_sensor']
        cam_ego_pose = info['cam_infos'][cam_key]['ego_pose']
        pts_img, features_img = map_pointcloud_to_image(
            pc.points.copy(), features.copy(), IMG_SHAPES[i], cam_calibrated_sensor, cam_ego_pose)

        file_name = os.path.split(info['cam_infos'][cam_key]['filename'])[-1]
        np.concatenate([pts_img[:2, :].T, features_img],
                       axis=1).astype(np.float32).flatten().tofile(
                           os.path.join(DATA_PATH, OUT_PATH,
                                        f'{file_name}.bin')) #(614,7)->(4298,)
    # plt.savefig(f"{sample_idx}")


def worker_fisheye(info, vis=False):
    radar_file_name = os.path.split(info['lidar_infos']['LIDAR_TOP']['filename'])[-1]
    points = np.fromfile(os.path.join(DATA_PATH, RADAR_SPLIT, radar_file_name),
                         dtype=np.float32,
                         count=-1).reshape(-1, MAX_DIM)

    lidar_calibrated_sensor = info['lidar_infos'][lidar_key][
        'calibrated_sensor']
    lidar_ego_pose = info['lidar_infos'][lidar_key]['ego_pose']

    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.
    pc = LidarPointCloud(points[:, :4].T)  # use 4 dims for code compatibility
    features = points[:, 3:] #(1795,7)->(1795, 4),rcs, vx_comp, vy_comp, (dummy field for sweep info)

    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor['translation']))

    # Second step: transform from ego to the global frame.
    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation']))

    for i, cam_key in enumerate(cam_keys):
        cam_calibrated_sensor = info['cam_infos'][cam_key]['calibrated_sensor']
        cam_ego_pose = info['cam_infos'][cam_key]['ego_pose']
        pts_img, features_img = map_pointcloud_to_fisheye_image(
            pc.points.copy(), features.copy(), IMG_SHAPES[i], cam_calibrated_sensor, cam_ego_pose, cam_key)

        file_name = os.path.split(info['cam_infos'][cam_key]['filename'])[-1]
        np.concatenate([pts_img[:2, :].T, features_img],
                       axis=1).astype(np.float32).flatten().tofile(
                           os.path.join(DATA_PATH, OUT_PATH,
                                        f'{file_name}.bin')) #(614,6)->(3684,)

        if vis:
            plot_on_image(os.path.join(DATA_PATH, info['cam_infos'][cam_key]['filename']), pts_img)
            
def worker_lidar_fisheye(info, vis=False):
    points = np.fromfile(os.path.join(DATA_PATH,info['lidar_infos']['LIDAR_TOP']['filename']),
                         dtype=np.float32,
                         count=-1).reshape(-1, 5)

    lidar_calibrated_sensor = info['lidar_infos'][lidar_key][
        'calibrated_sensor']
    lidar_ego_pose = info['lidar_infos'][lidar_key]['ego_pose']

    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.
    pc = LidarPointCloud(points[:, :4].T)  # use 4 dims for code compatibility
    features = points[:, 3:] #(1795,7)->(1795, 4),rcs, vx_comp, vy_comp, (dummy field for sweep info)

    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor['translation']))

    # Second step: transform from ego to the global frame.
    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation']))

    for i, cam_key in enumerate(cam_keys):
        cam_calibrated_sensor = info['cam_infos'][cam_key]['calibrated_sensor']
        cam_ego_pose = info['cam_infos'][cam_key]['ego_pose']
        pts_img, features_img = map_pointcloud_to_fisheye_image(
            pc.points.copy(), features.copy(), IMG_SHAPES[i], cam_calibrated_sensor, cam_ego_pose, cam_key)

        # file_name = os.path.split(info['cam_infos'][cam_key]['filename'])[-1]
        # np.concatenate([pts_img[:2, :].T, features_img],
        #                axis=1).astype(np.float32).flatten().tofile(
        #                    os.path.join(DATA_PATH, OUT_PATH,
        #                                 f'{file_name}.bin')) #(614,7)->(4298,)

        if vis:
            plot_on_image(os.path.join(DATA_PATH, info['cam_infos'][cam_key]['filename']), pts_img)
            

if __name__ == '__main__':
    mmcv.mkdir_or_exist(os.path.join(DATA_PATH, OUT_PATH))
    for info_path in info_paths:
        infos = mmcv.load(info_path)
        for info in tqdm(infos):
            # worker(info)
            worker_fisheye(info)
            # worker_lidar_fisheye(info, vis=True)
