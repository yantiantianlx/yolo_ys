import json
import cv2
import re
import numpy as np
from os.path import join
import os
import matplotlib.pyplot as plt


def split_extension_name(name):
    extension_pattern = re.compile('\\.[a-zA-z]*$')
    search = extension_pattern.search(name)
    assert search is not None, 'wrong file! at name: {}'.format(name)
    search_start = search.regs[0][0]
    return name[:search_start], name[search_start:]


# test =================================================================
test_root_dir = '/simple_ssd/ys2/data_clear/human_detection_data/test'
# train ================================================================
train_root_dir = '/simple_ssd/ys2/data_clear/human_detection_data/train'

root_dir = train_root_dir

image_dir = join(root_dir, 'image')
json_dir = join(root_dir, 'json')
for image_name in os.listdir(image_dir):
    abs_image_name = join(image_dir, image_name)
    name_ = split_extension_name(image_name)[0]
    abs_json_name = join(json_dir, name_+'.json')

    im = cv2.imread(abs_image_name)
    jsn = json.load(open(abs_json_name))

    im_w, im_h = jsn['image_size']['w'],  jsn['image_size']['h']

    color = (0, 255, 0)
    for human in jsn['human_list']:
        if 'is_crowd' in human:
            if human['is_crowd'] == 1:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)

        human_rect = human['human_rect']
        xmin = human_rect['xmin']
        xmax = human_rect['xmax']
        ymin = human_rect['ymin']
        ymax = human_rect['ymax']

        xmin = int(xmin*im_w)
        xmax = int(xmax*im_w)
        ymin = int(ymin*im_h)
        ymax = int(ymax*im_h)

        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), color, 2)

    # plt.imshow(im)
    # plt.show()

    # cv2.imshow(abs_image_name, im)
    cv2.imshow('', im)
    cv2.waitKey()
