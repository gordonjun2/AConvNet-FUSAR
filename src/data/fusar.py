from skimage.util import shape

import numpy as np
import tifffile as tiff
import tqdm

import glob
import os

import cv2

# target_name_fusar = ('Cargo', 'DiveVessel', 'Dredger', 'Fishing', 'HighSpeedCraft', 'LawEnforce', 'Other', 
#                      'Passenger', 'PortTender', 'Reserved', 'SAR', 'Tanker', 'Tug', 'Unspecified', 'WingInGrnd')

target_name_fusar = ('Cargo', 'Fishing', 'Tanker')

target_name = {
    'fusar': target_name_fusar
}

# serial_number = {
#     'Cargo': 0,

#     'DiveVessel': 1,
#     'Dredger': 2,
#     'Fishing': 3,

#     'HighSpeedCraft': 4,
#     'LawEnforce': 5,
#     'Other': 6,
#     'Passenger': 7,
#     'PortTender': 8,

#     'Reserved': 9,
#     'SAR': 10,
#     'Tanker': 11,
#     'Tug': 12,
#     'Unspecified': 13,
#     'WingInGrnd': 14
# }

serial_number = {
    'Cargo': 0,
    'Fishing': 1,
    'Tanker': 2
}


class FUSAR(object):

    def __init__(self, name='fusar', is_train=False, use_phase=False, chip_size=94, patch_size=88, stride=40):
        self.name = name
        self.is_train = is_train
        self.use_phase = use_phase
        self.chip_size = chip_size
        self.patch_size = patch_size
        self.stride = stride

    def read(self, path):

        _data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        h = _data.shape[0]
        w = _data.shape[1]

        _data = _data.reshape(-1, h, w)
        _data = _data.transpose(1, 2, 0)
        _data = _data.astype(np.float32)
        # if not self.use_phase:
        #     _data = np.expand_dims(_data[:, :, 0], axis=2)

        _data = self._normalize(_data)

        # _data = self._center_crop(_data)

        if self.is_train:
            _data = self._data_augmentation(_data, patch_size=self.patch_size, stride=self.stride)
        else:
            _data = [self._center_crop(_data, size=self.patch_size)]

        meta_label = self._extract_meta_label(path)
        return meta_label, _data

    @staticmethod
    def _center_crop(data, size=88):
        h, w, _ = data.shape

        y = (h - size) // 2
        x = (w - size) // 2

        return data[y: y + size, x: x + size]

    def _data_augmentation(self, data, patch_size=88, stride=40):
        # patch extraction
        _data = FUSAR._center_crop(data, size=self.chip_size)
        _, _, channels = _data.shape
        patches = shape.view_as_windows(_data, window_shape=(patch_size, patch_size, channels), step=stride)
        patches = patches.reshape(-1, patch_size, patch_size, channels)
        return patches

    def _extract_meta_label(self, path):

        path_split = path.split('\\')
        target_type = path_split[-2]
        class_id = target_name[self.name].index(target_type)

        return {
            'class_id': class_id,
            'target_type': target_type
        }

    @staticmethod
    def _get_azimuth_angle(angle):
        azimuth_angle = eval(angle)
        if azimuth_angle > 180:
            azimuth_angle -= 180
        return int(azimuth_angle)

    @staticmethod
    def _normalize(x):
        d = (x - x.min()) / (x.max() - x.min())
        return d.astype(np.float32)
