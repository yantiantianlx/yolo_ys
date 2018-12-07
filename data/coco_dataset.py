import os
import re
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import torch
from torch.utils.data import Dataset
from os.path import join

from gear_tool.conversion_utils import cv2_im_to_torch_im, torch_im_to_cv2_im, square_padding, x1y1x2y2_to_cxcywh
from data.plt_show_im_with_rect import plt_show
from data.coco_category import coco_cate_id_to_name, coco_cate_id_to_class_id, coco_class_to_name


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


class COCO_Dataset(Dataset):
    def __init__(self, arg, root_path, relative_path_txt=None, im_size=None, output_device='cpu'):
        self.arg = arg
        self.root_path = root_path
        self.model_input_wh = im_size
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

    def __getitem__(self, index):

        index = index % len(self.image_list)

        # read image and json ================================================================
        image_path = self.image_list[index]
        np_im = cv2.imread(image_path)
        im_h, im_w, im_c = np_im.shape
        np_im = cv2.cvtColor(np_im, cv2.COLOR_RGB2BGR)

        json_path = self.json_list[index]
        with open(json_path) as json_file:
            jsn = json.load(json_file)

        bbox_list = list()
        class_list = list()
        annotations = jsn['annotation']
        for anno in annotations:
            class_list.append(coco_cate_id_to_class_id(anno['category_id']))
            x, y, w, h = anno['bbox']
            flt_xmin = round(x/im_w, 6)
            flt_ymin = round(y/im_h, 6)
            flt_xmax = round((x+w)/im_w, 6)
            flt_ymax = round((y+h)/im_h, 6)
            bbox_list.append([flt_xmin, flt_ymin, flt_xmax, flt_ymax])

        if len(bbox_list) == 0:
            return self.__getitem__(index+1)

        # data pre process ===================================================================

        np_im, pad_hwc, bbox_list = square_padding(np_im, bbox_list=bbox_list)
        np_im = cv2.resize(np_im, (self.model_input_wh[0], self.model_input_wh[1]))

        # data output ========================================================================
        labels = np.zeros((self.max_objects, 5))
        for bbox_idx, bbox_rect in enumerate(bbox_list):
            if bbox_idx >= self.max_objects:
                break

            xmid_flt, ymid_flt, w_flt, h_flt = x1y1x2y2_to_cxcywh(bbox_rect)
            cls= class_list[bbox_idx]
            labels[bbox_idx, :] = cls, xmid_flt, ymid_flt, w_flt, h_flt

        # # # # # # # TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST
        # print(image_path)
        # np_im_show = cv2.cvtColor(np_im, cv2.COLOR_RGB2BGR).copy()
        # for bbox_idx, bbox_rect in enumerate(bbox_list):
        #     cls = class_list[bbox_idx]
        #     xmin_flt, ymin_flt, xmax_flt, ymax_flt = bbox_rect
        #     xmin, xmax, ymin, ymax = map(lambda x: int(x*self.model_input_wh[0]), [xmin_flt, xmax_flt, ymin_flt, ymax_flt])
        #     print(xmin, ymin, xmax, ymax)
        #     color = (0, 255, 0)
        #     cv2.rectangle(np_im_show, (xmin, ymin), (xmax, ymax), color, 2)
        #     cv2.putText(np_im_show, str(coco_class_to_name(cls)), (xmin, ymin+15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.imshow('resized_np_im_cp', np_im_show)
        # cv2.waitKey()
        # # # # # # # TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST

        data_dict = dict()
        data_dict['image'] = cv2_im_to_torch_im(np_im).to(self.output_device)
        data_dict['label'] = torch.from_numpy(labels).to(self.output_device)
        data_dict["image_path"] = image_path

        return data_dict

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    from gear_config.yolo_config.tiny_yolo_default import ARG
    arg = ARG()

    root_path = arg.data.dataset.train.root
    coco_dataset = COCO_Dataset(arg, root_path, im_size=arg.model.net.im_size)
    for i in coco_dataset:
        pass

    # root_path = arg.data.dataset.valid.root
    # hand_dataset = Hand_Dataset(arg, root_path, im_size=arg.model.net.im_size, mode='valid')

    # from torch.utils.data import DataLoader
    # train_loader = DataLoader(coco_dataset, batch_size=16, shuffle=False, drop_last=True)
    # for batch_idx, batch_data_dict in enumerate(train_loader):
    #     batch_label = batch_data_dict['label']

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

