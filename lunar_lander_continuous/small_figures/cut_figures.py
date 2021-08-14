import cv2

import os

import numpy as np

for _folder in os.listdir():
    if os.path.isdir(_folder):
        for img_name in os.listdir(_folder):
            if os.path.exists(os.path.join(_folder, img_name).replace('.png', '_cutted.png')):
                continue
            if img_name.endswith('_cutted.png'):
                continue

            if img_name.endswith('.png'):
                _img = cv2.imread(os.path.join(_folder, img_name))
                img = np.sum(_img, axis=2)

                for axis in range(2):
                    axis_mod = (axis + 1) % 2
                    img_axis = np.sum(img, axis=axis)
                    all_white = img.shape[axis] * 765

                    result = np.where(img_axis != all_white)[0]
                    _img = np.delete(_img, np.arange(result[-1], _img.shape[axis_mod]), axis_mod)
                    _img = np.delete(_img, np.arange(result[0]), axis_mod)

                cv2.imwrite(os.path.join(_folder, img_name).replace('.png', '_cutted.png'), _img)
