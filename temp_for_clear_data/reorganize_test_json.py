import re
import os
import json
import cv2
import shutil
from os.path import join


def split_extension_name(name):
    extension_pattern = re.compile('\\.[a-zA-z]*$')
    search = extension_pattern.search(name)
    assert search is not None, 'wrong file! at name: {}'.format(name)
    search_start = search.regs[0][0]
    return name[:search_start], name[search_start:]


# train ================================================================================================================
# from_path = '/simple_ssd/ys2/tiny_yolo_project/data/human_detection/20181101_merged'
# to_path = '/simple_ssd/ys2/data_clear/human_detection_data/train'
# to_path_image = join(to_path, 'image')
# to_path_json = join(to_path, 'json')


# test ================================================================================================================
# from_path = '/simple_ssd/ys2/human_test_images_1127'
# to_path = '/simple_ssd/ys2/data_clear'

from_path = '/home/ys/Desktop/human_test_images_1127'
to_path = '/home/ys/Desktop/test'
to_path_image = join(to_path, 'image')
to_path_json = join(to_path, 'json')

os.makedirs(to_path_image, exist_ok=True)
os.makedirs(to_path_json, exist_ok=True)

for name in os.listdir(from_path):
    if split_extension_name(name)[1] == '.json':
        json_path = os.path.join(from_path, name)
        image_path = split_extension_name(json_path)[0]+'.jpg'

        new_json_path = join(to_path_json, name)
        new_image_path = join(to_path_image, name)
        new_image_path = split_extension_name(new_image_path)[0] + '.jpg'

        new_jsn = dict()

        im = cv2.imread(image_path)
        im_height, im_width, im_channels = im.shape

        new_jsn['image_size'] = {'h': im_height, 'w': im_width, 'c': im_channels}

        jsn = json.load(open(json_path))
        new_jsn['image_name'] = jsn['human_name']

        new_human_list = list()
        for human in jsn['human_list']:
            new_human = dict()

            x = human['normalize_points'][0]
            y = human['normalize_points'][1]
            w = human['normalize_points'][2]
            h = human['normalize_points'][3]

            # x = human['human_rect']['x']
            # y = human['human_rect']['y']
            # w = human['human_rect']['w']
            # h = human['human_rect']['h']

            x, y, w, h = map(lambda t: int(t), (x, y, w, h))

            # print((x, y), (x+w, y+h))
            cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)

            xmin = x/im_width
            ymin = y/im_height
            xmax = (x+w)/im_width
            ymax = (y+h)/im_height

            new_human['human_rect'] = {'xmin': round(xmin, 6), 'ymin': round(ymin, 6), 'xmax': round(xmax, 6), 'ymax':round(ymax, 6)}

            xmin = int(xmin*im_width)
            ymin = int(ymin*im_height)
            xmax = int(xmax*im_width)
            ymax = int(ymax*im_height)
            # print((xmin, ymin), (xmax, ymax))
            cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (255, 255, 0), 1)

            # new_keypoint_list = list()
            # for keypoint in human['human_keypoints']:
            #     x = keypoint['x']
            #     y = keypoint['y']
            #     is_v = keypoint['is_visible']
            #     conf = keypoint['confidence']
            #     new_keypoint = {'x': round(x/im_width, 6), 'y': round(y/im_height, 6), 'is_visible':is_v, 'confi':conf}
            #     new_keypoint_list.append(new_keypoint)
            #
            # new_human['human_keypoint_list'] = new_keypoint_list

            new_human_list.append(new_human)
        new_jsn['human_list'] = new_human_list

        new_human_no_mark_list = list()
        # for human in jsn['human_not_marked']:
        #     new_human = dict()
        #
        #     x = human['human_rect']['x']
        #     y = human['human_rect']['y']
        #     w = human['human_rect']['w']
        #     h = human['human_rect']['h']
        #
        #     xmin = x/im_width
        #     ymin = y/im_height
        #     xmax = (x+w)/im_width
        #     ymax = (y+h)/im_width
        #
        #     new_human['human_rect'] = {'xmin': round(xmin, 6), 'ymin': round(ymin, 6), 'xmax': round(xmax, 6),'ymax': round(ymax, 6)}
        #
        #     new_human_no_mark_list.append(new_human)
        # new_jsn['human_no_mark_list'] = new_human_no_mark_list

        cv2.imshow('', im)
        cv2.waitKey()

        # with open(new_json_path, 'x') as fp:
        #     json.dump(new_jsn, fp)
        #
        # shutil.copy(image_path, new_image_path)
