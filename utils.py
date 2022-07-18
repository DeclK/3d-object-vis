import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import cv2 as cv

# 层级打印一个字典
def print_dict(dict_, content=False, level=0):
    for k, v in dict_.items():
        print('\t' * level + f'{k}')
        if type(v) == dict:
            print_dict(v, content, level + 1)
        elif content: print(v, '\n', '-' * 20)

def get_calib(calib_path):
    """
    Get calib dict from calib.txt.
    Please download KITTI raw's calibration file, which includes calib_cam_to_cam.txt and
    calib_velo_to_cam.txt. Please put these two files into one, and name it as calib.txt
    """
    text = calib_path.read_text()
    calib = {}
    for lines in text.split('\n'):
        label, data = lines.split(':')
        data = data.split(' ')[1:]
        data = [float(i) for i in data]
        data = np.array(data)
        if label == 'R_rect_00':
            data = data.reshape((3, 3))
            data_ = np.zeros((4, 4))
            data_[3, 3] = 1.0
            data_[:3, :3] = data
            calib['R0_rect'] = data_
        elif label == 'P_rect_02':
            data = data.reshape((3, 4))
            data_ = np.zeros((4, 4))
            data_[3, 3] = 1.0
            data_[:3, :4] = data
            calib['P2'] = data_
        elif label == 'R':
            data = data.reshape((3, 3))
            calib['Tr_velo_to_cam'] = data
        else:
            data = data.reshape((3, 1))
            data = np.concatenate((calib['Tr_velo_to_cam'], data), axis=1)
            data_ = np.zeros((4, 4))
            data_[3, 3] = 1.0
            data_[:3, :4] = data
            calib['Tr_velo_to_cam'] = data_
    return calib

def concat(data_root:Path):
    """
    Concat velodyne camera and RGB camera image
    the dir should arange like this:
        - data_root
            - img
            - velo
            - (optional) concat: the result will be saved here
    """
    img_path = data_root / 'img'
    velo_path = data_root / 'velo'
    concat_path = data_root / 'concat'
    concat_path.mkdir(parents=True, exist_ok=True)

    img_list = os.listdir(img_path)
    velo_list = os.listdir(velo_path)
    pbar = tqdm(range(len(img_list)))
    pbar.set_description('concat')
    for idx in pbar:
        camera = cv.imread(str(img_path / img_list[idx]))
        velo = cv.imread(str(velo_path / velo_list[idx]))
        factor = velo.shape[1] / camera.shape[1]
        camera = cv.resize(camera, dsize=None, fx=factor, fy=factor)
        velo[:camera.shape[0],:camera.shape[1]] = camera
        cv.imwrite(str(concat_path / img_list[idx]), velo)