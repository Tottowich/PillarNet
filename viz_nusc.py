import numpy as np
import json
import matplotlib.pyplot as plt
from typing import List, Tuple
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix


nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuScenes',
                verbose=True)


def format_boxes(sd_record, records, box_vis_level: BoxVisibility = BoxVisibility.ANY,
          selected_anntokens: List[str] = None,
          use_flat_vehicle_coordinates: bool = True):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_anntokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """
    # Retrieve sensor & pose records
    # sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    boxes = []
    for record in records:
        box = Box(record['translation'], record['size'], Quaternion(record['rotation']),
                  name=record['detection_name'], token=record['sample_token'])
        boxes.append(box)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue

        box_list.append(box)
    return box_list


def get_color(category_name: str) -> Tuple[int, int, int]:
    """
    Provides the default colors based on the category names.
    This method works for the general nuScenes categories, as well as the nuScenes detection categories.
    """
    if 'bicycle' in category_name or 'motorcycle' in category_name:
        return 255, 61, 99  # Red
    elif 'vehicle' in category_name or category_name in ['bus', 'car', 'construction_vehicle', 'trailer', 'truck']:
        return 255, 158, 0  # Orange
    elif 'pedestrian' in category_name:
        return 0, 0, 230  # Blue
    elif 'cone' in category_name or 'barrier' in category_name:
        return 0, 0, 0  # Black
    else:
        return 255, 0, 255  # Magenta


sensor = 'LIDAR_TOP'


def save_video(method, val_tokens, out_path, start=0, end=0):
    for idx, sample_token in enumerate(val_tokens[start:end]):
        idx = idx + start
        print(idx)
        cur_sample = nusc.get('sample', sample_token)
        # gt_data = cur_sample['data']

        lidar_top_data = nusc.get('sample_data', cur_sample['data'][sensor])
        ax = nusc.render_sample_data(lidar_top_data['token'], with_anns=False, axes_limit=54)

        records = method[sample_token]
        boxes = format_boxes(lidar_top_data, records)

        for box in boxes:
            c = np.array(get_color(box.name)) / 255.0
            box.render(ax, view=np.eye(4), colors=(c, c, c))

        # nusc.render_sample_data(lidar_top_data['token'], with_anns=True, axes_limit=54)
        plt.savefig(out_path + f"{idx}.png", dpi=300)
        plt.close('all')


def save_gt_video(val_tokens, out_path):
    for idx, sample_token in enumerate(val_tokens):
        print(idx)
        cur_sample = nusc.get('sample', sample_token)
        # gt_data = cur_sample['data']

        lidar_top_data = nusc.get('sample_data', cur_sample['data'][sensor])
        nusc.render_sample_data(lidar_top_data['token'], with_anns=True, axes_limit=54)
        plt.savefig(out_path + f"{idx}.png", dpi=300)
        plt.close('all')


if __name__ == '__main__':
    with open("pointpillars.json", "r") as f:
        pointpillars = json.load(f)
        pointpillars = pointpillars["results"]

    with open("second.json", "r") as f:
        second = json.load(f)
        second = second["results"]

    with open("pillarnet.json", "r") as f:
        pillarnet = json.load(f)
        pillarnet = pillarnet["results"]

    val_tokens = pointpillars.keys()
    val_tokens = list(val_tokens)
    # val_tokens.sort()

    # nusc.list_scenes()
    # sample_tokens = nusc.sample

    # save_gt_video(val_tokens, out_path="/home/sgs/Pictures/gtval/")
    # save_video(pointpillars, val_tokens, "/home/sgs/Pictures/pointpillars/", 240, 320)
    save_video(second, val_tokens, "/home/sgs/Pictures/second/", 240, 320)
    # save_video(pillarnet, val_tokens, "/home/sgs/Pictures/pillarnet/", 240, 320)
