
import yaml
from collections import OrderedDict
import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa
# import intervals as I


def yaml_ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


# yaml_path = '/simple_ssd/ys2/tiny_yolo_project/tiny_yolo/gear_config/yolo_config/body_micro_yolo_v1_default.yaml'
# a = yaml_ordered_load(open(yaml_path))
# print()

class Augmenter:
    def __init__(self, float_coordinate=True):
        self.float_coordinate = float_coordinate
        self.augment_dict = OrderedDict()
        self.seq = None
        self.set_seq()

    def dict_to_sequential(self):
        for k, v in self.augment_dict.items():
            pass

    def set_seq(self, seq=None):
        if seq is not None:
            self.seq = seq

    def process(self, image_list, point_list=None, bbox_list=None):
        list_flag = True
        if isinstance(image_list, np.ndarray):
            list_flag = False
            image_list = [image_list]

        assert len(image_list) > 0

        # handle image ============================================================================
        seq_det = self.seq.to_deterministic()

        image_aug_list = list()
        for image in image_list:
            image_aug = seq_det.augment_images([image])[0]
            image_aug_list.append(image_aug)

        res_list = [image_aug_list] if list_flag else [image_aug_list[0]]

        # handle point list ========================================================================
        whole_point_list = list()
        point_list_len = 0
        if point_list is not None:
            point_list_len = len(point_list)
            whole_point_list = point_list
        if bbox_list is not None:
            for bbox in bbox_list:
                x1, y1, x2, y2 = bbox
                whole_point_list.append([x1, y1])
                whole_point_list.append([x2, y2])

        im_h, im_w, im_c = image_list[0].shape
        if len(whole_point_list):
            ia_point_list = list()
            for point in whole_point_list:
                iax = point[0]
                iay = point[1]
                if self.float_coordinate:
                    iax = min(max(iax, 0.000), 1.000)
                    iay = min(max(iay, 0.000), 1.000)
                    iax *= im_w
                    iay *= im_h
                ia_point_list.append(ia.Keypoint(x=iax, y=iay))

            keypoints = ia.KeypointsOnImage(ia_point_list, shape=image_list[0].shape)
            keypoints_aug = seq_det.augment_keypoints([keypoints])[0]

            new_whole_point_list = list()
            for point in keypoints_aug.keypoints:
                if self.float_coordinate:
                    new_whole_point_list.append([point.x / im_w, point.y / im_h])
                else:
                    new_whole_point_list.append([point.x_int, point.y_int])

            # rearrange the output.
            new_point_list = new_whole_point_list[:point_list_len]
            if len(new_point_list):
                res_list.append(new_point_list)

            new_rect_point_list = new_whole_point_list[point_list_len:]
            if len(new_rect_point_list):
                new_rect_list = list()
                for i in range(len(new_rect_point_list) // 2):
                    even_idx = 2 * i
                    odd_idx = 2 * i + 1
                    x1, y1 = new_rect_point_list[even_idx]
                    x2, y2 = new_rect_point_list[odd_idx]
                    new_rect_list.append([x1, y1, x2, y2])

                # for flip xmin, xmax and ymin, ymax may exchange
                for rect in new_rect_list:
                    if rect[0] > rect[2]:
                        rect[0], rect[2] = rect[2], rect[0]
                    if rect[1] > rect[3]:
                        rect[1], rect[3] = rect[3], rect[1]

                res_list.append(new_rect_list)

        return res_list


if __name__ == '__main__':
    # import numpy as np
    # import cv2
    # image = cv2.imread('../tst/images/t2.jpg')
    # auger = Augmenter()
    # image_list, keypoint_list = auger.process([image], [[0.5, 0.5], [0.6, 0.6]])
    # print()
    pass
