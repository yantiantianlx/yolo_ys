import os
import re
import numpy as np
import cv2
import json
import torch
from torch.utils.data import Dataset
from os.path import join

from imgaug import augmenters as iaa
from data.augmentation import Augmenter
from gear_tool.conversion_utils import cv2_im_to_torch_im, torch_im_to_cv2_im, square_padding, x1y1x2y2_to_cxcywh
from data.rectangle_select import get_scale_dx_dy


def corresponding_json_path(image_path):
    file_, extension = split_extension_name(image_path)
    json_path = file_ + '.json'
    return json_path


def split_extension_name(name):
    extension_pattern = re.compile('\\.[a-zA-z]*$')
    search = extension_pattern.search(name)
    assert search is not None, 'wrong file! at name: {}'.format(name)
    search_start = search.regs[0][0]
    return name[:search_start], name[search_start:]


class Hand_Dataset(Dataset):
    def __init__(self, arg, root_path, relative_path_txt=None, im_size=None, mode='train', output_device='cpu'):
        self.arg = arg
        self.root_path = root_path
        self.model_input_wh = im_size
        self.mode = mode
        self.output_device = output_device
        self.image_list = []
        self.json_list = []
        self.max_objects = arg.data.dataset.train.max_detection_num
        if relative_path_txt is not None:
            with open(relative_path_txt, 'r') as file:
                lines = file.readlines()
            for line in lines:
                image_path, json_path = line.split()
                abs_image_name = join(root_path, image_path)
                abs_json_name = join(root_path, json_path)

                self.image_list.append(abs_image_name)
                self.json_list.append(abs_json_name)
        else:
            image_dir = join(root_path, 'image')
            json_dir = join(root_path, 'json')
            for image_name in os.listdir(image_dir):
                abs_image_name = join(image_dir, image_name)
                name_ = split_extension_name(image_name)[0]
                abs_json_name = join(json_dir, name_ + '.json')

                self.image_list.append(abs_image_name)
                self.json_list.append(abs_json_name)

        self.augmenter = Augmenter()

    def filter_bbox(self, human_list):
        cleared_human_list = list()
        for human in human_list:
            x1, y1, x2, y2 = human
            w = x2 - x1
            h = y2 - y1
            # if w < self.arg.data.dataset.train.min_w:
            #     continue
            # if h < self.arg.data.dataset.train.min_h:
            #     continue

            if w*h < 0.01:
                continue

            if h / w > 10:
                continue
            if w / h > 2:
                continue

            cleared_human_list.append(human)

        return cleared_human_list

    @staticmethod
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

    @staticmethod
    def temp_data_arrange(human_list):
        for human in human_list:
            for i in range(4):
                if human[i] > 1.0:
                    human[i] = 1.0
                if human[i] < 0.0:
                    human[i] = 0.0
        return human_list

    @staticmethod
    def get_augmentation(mode, scale=1.0, dx=0, dy=0):
        if mode == 0:
            augmentation_sequential = iaa.Sequential([])
        elif mode == 1:
            augmentation_sequential = iaa.Sequential([
                iaa.Affine(
                    translate_percent={'x': dx, 'y': dy},
                    scale=scale
                )
            ])
        elif mode == 2:
            augmentation_sequential = iaa.Sequential([
                iaa.OneOf([iaa.Add((-20, 20), per_channel=0.5), iaa.Multiply((0.9, 1.1), per_channel=0.5)]),
                iaa.OneOf([iaa.GaussianBlur((0, 0.75)), iaa.AverageBlur((1, 3)), iaa.MedianBlur((1, 3))]),
                # iaa.Dropout((0, 0.01), per_channel=0.5),
                # iaa.CoarseDropout(0.01, size_percent=0.4),
                iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5),
                iaa.Grayscale((0.0, 1.0)),
                iaa.Affine(
                    translate_percent={'x': dx, 'y': dy},
                    scale=scale
                ),
                iaa.Fliplr(0.5),
            ])
        else:
            raise Exception('augmentation mode wrong!')
        return augmentation_sequential

    def __getitem__(self, index):

        index = index % len(self.image_list)

        # read image and json ================================================================
        image_path = self.image_list[index]
        np_im = cv2.imread(image_path)
        np_im = cv2.cvtColor(np_im, cv2.COLOR_RGB2BGR)

        json_path = self.json_list[index]
        with open(json_path) as json_file:
            jsn = json.load(json_file)

        human_list = jsn['human_list']
        human_list = self.dict_list_to_list_list(human_list)
        human_list = self.temp_data_arrange(human_list)

        # data pre process ===================================================================
        np_im, pad_hwc, human_list = square_padding(np_im, bbox_list=human_list)
        np_im = cv2.resize(np_im, (self.model_input_wh[0], self.model_input_wh[1]))

        # data augment =======================================================================
        if self.mode == 'train':
            human_list = self.filter_bbox(human_list)
            if len(human_list):
                scale, dx, dy = get_scale_dx_dy(human_list, self.arg.data.dataset.train.min_w, self.arg.data.dataset.train.min_h)
                mode = self.arg.data.dataset.augmentation
                augmentation_sequential = self.get_augmentation(mode, scale, dx, dy)
                self.augmenter.set_seq(augmentation_sequential)
                np_im, human_list = self.augmenter.process(np_im, bbox_list=human_list)

        # data output ========================================================================
        labels = np.zeros((self.max_objects, 5))
        for human_idx, human_rect in enumerate(human_list):
            if human_idx >= self.max_objects:
                break

            xmid_flt, ymid_flt, w_flt, h_flt = x1y1x2y2_to_cxcywh(human_rect)
            cls_idx = 0
            labels[human_idx, :] = cls_idx, xmid_flt, ymid_flt, w_flt, h_flt

        # # # # # # TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST
        # print(image_path)
        # np_im_show = np_im.copy()
        # for human_idx, human_rect in enumerate(human_list):
        #     xmin_flt, ymin_flt, xmax_flt, ymax_flt = human_rect
        #     xmin, xmax, ymin, ymax = map(lambda x: int(x*self.model_input_wh[0]), [xmin_flt, xmax_flt, ymin_flt, ymax_flt])
        #     print(xmin, ymin, xmax, ymax)
        #     color = (0, 255, 0)
        #     cv2.rectangle(np_im_show, (xmin, ymin), (xmax, ymax), color, 2)
        #
        #     cv2.circle(np_im_show, (xmin, ymin), 10, (255, 0, 0), 3)
        #     cv2.circle(np_im_show, (xmax, ymax), 10, (255, 0, 255), 3)
        # cv2.imshow('resized_np_im_cp', np_im_show)
        # cv2.waitKey()
        # # # # # # TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST

        data_dict = dict()
        data_dict['image'] = cv2_im_to_torch_im(np_im).to(self.output_device)
        data_dict['label'] = torch.from_numpy(labels).to(self.output_device)
        data_dict["image_path"] = image_path

        return data_dict

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    from gear_config.yolo_config.body_micro_yolo_aug1 import ARG
    arg = ARG()

    root_path = arg.data.dataset.train.root
    hand_dataset = Hand_Dataset(arg, root_path, relative_path_txt=arg.data.dataset.train.txt_path, im_size=arg.model.net.im_size, mode='train')

    # root_path = arg.data.dataset.valid.root
    # hand_dataset = Hand_Dataset(arg, root_path, im_size=arg.model.net.im_size, mode='valid')

    from torch.utils.data import DataLoader
    train_loader = DataLoader(hand_dataset, batch_size=16, shuffle=False, drop_last=True)
    for batch_idx, batch_data_dict in enumerate(train_loader):
        batch_label = batch_data_dict['label']
        bs = batch_label.shape[0]
        assert bs == 16
        print(bs)
    #
    # image_path = '/simple_ssd/ys2/human_detection_data/test/image/12218.jpg'
    # json_path = '/simple_ssd/ys2/human_detection_data/test/json/12218.json'
    #
    # im = cv2.imread(image_path)
    # im_h, im_w, _ = im.shape
    # # cv2.imshow('', im)
    # # cv2.waitKey()
    #
    # jsn = json.load(open(json_path))
    # print(jsn)
    # human_list = jsn['human_list']
    # for human in human_list:
    #     human_rect = human['human_rect']
    #     xmin, xmax, ymin, ymax = human_rect['xmin'], human_rect['xmax'], human_rect['ymin'], human_rect['ymax']
    #     xmin, xmax, ymin, ymax = map(int, [im_w * xmin, im_w * xmax, im_h * ymin, im_h * ymax])
    #     color = (0, 255, 0)
    #     cv2.rectangle(im, (xmin, ymin), (xmax, ymax), color, 2)
    #
    # cv2.imshow('resized_np_im_cp', im)
    # cv2.waitKey()

