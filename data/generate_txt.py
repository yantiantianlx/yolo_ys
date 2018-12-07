import os
import re
import json
from os.path import join
from tqdm import tqdm
import numpy as np


def bbox_iou_x1y1x2y2(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    # Intersection area
    inter_area = max(inter_rect_x2 - inter_rect_x1 + 1, 0) * \
                 max(inter_rect_y2 - inter_rect_y1 + 1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def split_extension_name(name):
    extension_pattern = re.compile('\\.[a-zA-z]*$')
    search = extension_pattern.search(name)
    assert search is not None, 'wrong file! at name: {}'.format(name)
    search_start = search.regs[0][0]
    return name[:search_start], name[search_start:]


def dict_list_to_list_list(human_list):
    new_human_list = list()
    for human in human_list:
        human_rect = human['human_rect']
        xmin = human_rect['xmin']
        xmax = human_rect['xmax']
        ymin = human_rect['ymin']
        ymax = human_rect['ymax']

        new_human_list.append([xmin, ymin, xmax, ymax])
    return new_human_list


def filter_json(arg, jsn):
    cleared_body_list = list()
    last_rect = None
    if len(jsn['human_list']) > 5:
        return None

    image_w = jsn['image_size']['w']
    image_h = jsn['image_size']['h']
    image_c = jsn['image_size']['c']
    for human in jsn['human_list']:
        if 'is_crowd' in human:
            if human['is_crowd'] == 1:
                return None

        xmin = human['human_rect']['xmin']
        ymin = human['human_rect']['ymin']
        xmax = human['human_rect']['xmax']
        ymax = human['human_rect']['ymax']

        dim_diff = np.abs(image_w - image_h)
        padded_h = max(image_w, image_h)
        pad1, pad2 = dim_diff // 2, dim_diff // 2
        pad_hwc = ((pad1, pad2), (0, 0), (0, 0)) if image_h <= image_w else ((0, 0), (pad1, pad2), (0, 0))  # pad to h / pad to w

        xmin = int(xmin * image_w)
        xmax = int(xmax * image_w)
        ymin = int(ymin * image_h)
        ymax = int(ymax * image_h)

        xmin = pad_hwc[1][0] + xmin
        xmax = pad_hwc[1][0] + xmax
        ymin = pad_hwc[0][0] + ymin
        ymax = pad_hwc[0][0] + ymax

        xmin = xmin / padded_h
        xmax = xmax / padded_h
        ymin = ymin / padded_h
        ymax = ymax / padded_h

        w = xmax - xmin
        h = ymax - ymin
        # if w < arg.data.dataset.train.min_w:
        #     continue
        # if h < arg.data.dataset.train.min_h:
        #     continue

        if w*h < 0.01:
            continue

        if h / w > 10:
            continue
        if w / h > 2:
            continue

        rect = [xmin, ymin, xmax, ymax]
        if last_rect is not None:
            if bbox_iou_x1y1x2y2(rect, last_rect) > 0.5:
                return None
        last_rect = rect

        cleared_body_list.append([xmin, xmax, ymin, ymax])
    if len(cleared_body_list) >= 2:
        return None
    if len(cleared_body_list) == 0:
        return None
    return json


if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(os.getcwd()))
    from gear_config.yolo_config.body_micro_yolo_aug1 import ARG

    arg = ARG()

    # train ===================================================================================================
    train_root_dir = arg.data.dataset.train.root

    data_root = train_root_dir
    image_dir = join(data_root, 'image')
    json_dir = join(data_root, 'json')
    txt_path = join(data_root, 'pair.txt')

    with open(txt_path, "w") as f:
        counter = 0
        for image_name in tqdm(os.listdir(image_dir)):
            rela_image_name = join('image', image_name)
            name_ = split_extension_name(image_name)[0]
            rela_json_name = join('json', name_ + '.json')

            abs_image_name = join(image_dir, image_name)
            name_ = split_extension_name(image_name)[0]
            abs_json_name = join(json_dir, name_ + '.json')

            # =========================================================================================
            jsn = json.load(open(abs_json_name))

            if filter_json(arg, jsn) is not None:
                counter += 1
                f.write(rela_image_name+' '+rela_json_name+'\n')
        print(counter)

