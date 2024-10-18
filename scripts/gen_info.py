import os
import mmcv
import numpy as np
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits


def generate_info(nusc, scenes, max_cam_sweeps=6, max_lidar_sweeps=10, max_radar_sweeps=6):
    infos = list()
    for cur_scene in tqdm(nusc.scene):
        # if cur_scene['name'] not in scenes:
        #     continue
        first_sample_token = cur_scene['first_sample_token']
        cur_sample = nusc.get('sample', first_sample_token)
        while True:
            info = dict()
            cam_datas = list()
            radar_datas = list()
            lidar_datas = list()
            info['scene_name'] = nusc.get('scene', cur_scene['token'])['name']
            info['sample_token'] = cur_sample['token']
            info['timestamp'] = cur_sample['timestamp']
            info['scene_token'] = cur_sample['scene_token']
            cam_names = [
                'CAM_FRONT', 
                'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT', 
                'CAM_AVM_FRONT', 'CAM_AVM_REAR', 'CAM_AVM_LEFT', 'CAM_AVM_RIGHT'
            ]
            radar_names = ["RADAR_BACK_RIGHT", "RADAR_BACK_LEFT", "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT"]
            lidar_names = ['LIDAR_TOP']
            cam_infos = dict()
            lidar_infos = dict()
            radar_infos = dict()
            for cam_name in cam_names:
                cam_data = nusc.get('sample_data',
                                    cur_sample['data'][cam_name])
                cam_datas.append(cam_data)
                sweep_cam_info = dict()
                sweep_cam_info['sample_token'] = cam_data['sample_token']
                sweep_cam_info['ego_pose'] = nusc.get(
                    'ego_pose', cam_data['ego_pose_token'])
                sweep_cam_info['timestamp'] = cam_data['timestamp']
                sweep_cam_info['is_key_frame'] = cam_data['is_key_frame']
                sweep_cam_info['height'] = cam_data['height']
                sweep_cam_info['width'] = cam_data['width']
                img_path = os.path.join('/defaultShare/tmpnfs/dataset/zm_radar/nuscenes_fmt_with_labels/24-09-20_00-00-01_000_test/',
                                        cam_data['filename'])
                if not os.path.exists(img_path):
                    print(f"{img_path} not exists")
                    # exit(1)
                sweep_cam_info['filename'] = cam_data['filename']
                sweep_cam_info['calibrated_sensor'] = nusc.get(
                    'calibrated_sensor', cam_data['calibrated_sensor_token'])
                cam_infos[cam_name] = sweep_cam_info
            for radar_name in radar_names:
                radar_data = nusc.get('sample_data',
                                      cur_sample['data'][radar_name])
                radar_datas.append(radar_data)
                sweep_radar_info = dict()
                sweep_radar_info['sample_token'] = radar_data['sample_token']
                sweep_radar_info['ego_pose'] = nusc.get(
                    'ego_pose', radar_data['ego_pose_token'])
                sweep_radar_info['is_key_frame'] = radar_data['is_key_frame']
                sweep_radar_info['prev'] = radar_data['prev']
                sweep_radar_info['timestamp'] = radar_data['timestamp']
                sweep_radar_info['filename'] = radar_data['filename']
                sweep_radar_info['calibrated_sensor'] = nusc.get(
                    'calibrated_sensor', radar_data['calibrated_sensor_token'])
                radar_infos[radar_name] = sweep_radar_info
            assert len(radar_infos) == 5                
            for lidar_name in lidar_names:
                lidar_data = nusc.get('sample_data',
                                      cur_sample['data'][lidar_name])
                lidar_datas.append(lidar_data)
                sweep_lidar_info = dict()
                sweep_lidar_info['sample_token'] = lidar_data['sample_token']
                sweep_lidar_info['ego_pose'] = nusc.get(
                    'ego_pose', lidar_data['ego_pose_token'])
                sweep_lidar_info['timestamp'] = lidar_data['timestamp']
                sweep_lidar_info['filename'] = lidar_data['filename']
                sweep_lidar_info['calibrated_sensor'] = nusc.get(
                    'calibrated_sensor', lidar_data['calibrated_sensor_token'])
                lidar_infos[lidar_name] = sweep_lidar_info

            lidar_sweeps = [dict() for _ in range(max_lidar_sweeps)]
            radar_sweeps = [dict() for _ in range(max_radar_sweeps)]
            cam_sweeps = [dict() for _ in range(max_cam_sweeps)]
            info['cam_infos'] = cam_infos
            info['radar_infos'] = radar_infos
            info['lidar_infos'] = lidar_infos
            for k, cam_data in enumerate(cam_datas):
                sweep_cam_data = cam_data
                for j in range(max_cam_sweeps):
                    if sweep_cam_data['prev'] == '':
                        break
                    else:
                        sweep_cam_data = nusc.get('sample_data',
                                                  sweep_cam_data['prev'])
                        sweep_cam_info = dict()
                        sweep_cam_info['sample_token'] = sweep_cam_data[
                            'sample_token']
                        if sweep_cam_info['sample_token'] != cam_data[
                                'sample_token']:
                            break
                        sweep_cam_info['ego_pose'] = nusc.get(
                            'ego_pose', cam_data['ego_pose_token'])
                        sweep_cam_info['timestamp'] = sweep_cam_data[
                            'timestamp']
                        sweep_cam_info['is_key_frame'] = sweep_cam_data[
                            'is_key_frame']
                        sweep_cam_info['height'] = sweep_cam_data['height']
                        sweep_cam_info['width'] = sweep_cam_data['width']
                        sweep_cam_info['filename'] = sweep_cam_data['filename']
                        sweep_cam_info['calibrated_sensor'] = nusc.get(
                            'calibrated_sensor',
                            cam_data['calibrated_sensor_token'])
                        cam_sweeps[j][cam_names[k]] = sweep_cam_info

            for k, radar_data in enumerate(radar_datas):
                sweep_radar_data = radar_data
                for j in range(max_radar_sweeps):
                    if sweep_radar_data['prev'] == '':
                        break
                    else:
                        sweep_radar_data = nusc.get('sample_data',
                                                    sweep_radar_data['prev'])
                        sweep_radar_info = dict()
                        sweep_radar_info['sample_token'] = sweep_radar_data[
                            'sample_token']
                        if sweep_radar_info['sample_token'] != radar_data[
                                'sample_token']:
                            break
                        sweep_radar_info['ego_pose'] = nusc.get(
                            'ego_pose', sweep_radar_data['ego_pose_token'])
                        sweep_radar_info['timestamp'] = sweep_radar_data[
                            'timestamp']
                        sweep_radar_info['is_key_frame'] = sweep_radar_data[
                            'is_key_frame']
                        sweep_radar_info['filename'] = sweep_radar_data[
                            'filename']
                        sweep_radar_info['calibrated_sensor'] = nusc.get(
                            'calibrated_sensor',
                            cam_data['calibrated_sensor_token'])
                        radar_sweeps[j][radar_names[k]] = sweep_radar_info
                        # if list(radar_sweeps[j].keys())[0] != 'RADAR_BACK_RIGHT':
                        #     print("error")                        
                        
            for k, lidar_data in enumerate(lidar_datas):
                sweep_lidar_data = lidar_data
                for j in range(max_lidar_sweeps):
                    if sweep_lidar_data['prev'] == '':
                        break
                    else:
                        sweep_lidar_data = nusc.get('sample_data',
                                                    sweep_lidar_data['prev'])
                        sweep_lidar_info = dict()
                        sweep_lidar_info['sample_token'] = sweep_lidar_data[
                            'sample_token']
                        if sweep_lidar_info['sample_token'] != lidar_data[
                                'sample_token']:
                            break
                        sweep_lidar_info['ego_pose'] = nusc.get(
                            'ego_pose', sweep_lidar_data['ego_pose_token'])
                        sweep_lidar_info['timestamp'] = sweep_lidar_data[
                            'timestamp']
                        sweep_lidar_info['is_key_frame'] = sweep_lidar_data[
                            'is_key_frame']
                        sweep_lidar_info['filename'] = sweep_lidar_data[
                            'filename']
                        sweep_lidar_info['calibrated_sensor'] = nusc.get(
                            'calibrated_sensor',
                            cam_data['calibrated_sensor_token'])
                        lidar_sweeps[j][lidar_names[k]] = sweep_lidar_info
            # Remove empty sweeps.
            for i, sweep in enumerate(cam_sweeps):
                if len(sweep.keys()) == 0:
                    cam_sweeps = cam_sweeps[:i]
                    break
            for i, sweep in enumerate(radar_sweeps):
                try:
                    assert len(sweep) == 5 or len(sweep) == 0
                except:
                    print("error")
                if len(sweep.keys()) == 0:
                    radar_sweeps = radar_sweeps[:i]
                    break
            for i, sweep in enumerate(lidar_sweeps):
                if len(sweep.keys()) == 0:
                    lidar_sweeps = lidar_sweeps[:i]
                    break
            info['cam_sweeps'] = cam_sweeps
            info['lidar_sweeps'] = lidar_sweeps
            info['radar_sweeps'] = radar_sweeps
            ann_infos = list()

            if 'anns' in cur_sample:
                for ann in cur_sample['anns']:
                    ann_info = nusc.get('sample_annotation', ann)
                    velocity = nusc.box_velocity(ann_info['token'])
                    if np.any(np.isnan(velocity)):
                        velocity = np.zeros(3)
                    ann_info['velocity'] = velocity
                    ann_infos.append(ann_info)
                info['ann_infos'] = ann_infos
            infos.append(info)
            if cur_sample['next'] == '':
                break
            else:
                cur_sample = nusc.get('sample', cur_sample['next'])
    return infos


def main():
    trainval_nusc = NuScenes(version='v1.0-trainval',
                             dataroot='/defaultShare/tmpnfs/dataset/zm_radar/nuscenes_fmt_with_labels/24-09-20_00-00-01_000_test/',
                             verbose=True)
    train_scenes = splits.train
    val_scenes = splits.val
    # train_infos_tiny = generate_info(trainval_nusc, train_scenes[:2])
    # mmcv.dump(train_infos_tiny, '/defaultShare/tmpnfs/dataset/zm_radar/nuscenes_fmt_with_labels/24-09-04_2/nuscenes_infos_train-tiny.pkl')
    train_infos = generate_info(trainval_nusc, train_scenes)
    mmcv.dump(train_infos, '/defaultShare/tmpnfs/dataset/zm_radar/nuscenes_fmt_with_labels/24-09-20_00-00-01_000_test/nuscenes_infos_train.pkl')
    val_infos = generate_info(trainval_nusc, val_scenes)
    mmcv.dump(val_infos, '/defaultShare/tmpnfs/dataset/zm_radar/nuscenes_fmt_with_labels/24-09-20_00-00-01_000_test/nuscenes_infos_val.pkl')

    # test_nusc = NuScenes(version='v1.0-test',
    #                      dataroot='/defaultShare/tmpnfs/dataset/zm_radar/nuscenes_gen/24-09-04_2/',
    #                      verbose=True)
    # test_scenes = splits.test
    # test_infos = generate_info(test_nusc, test_scenes)
    # mmcv.dump(test_infos, '/defaultShare/tmpnfs/dataset/zm_radar/nuscenes_gen/24-09-04_2/nuscenes_infos_test.pkl')


if __name__ == '__main__':
    main()
