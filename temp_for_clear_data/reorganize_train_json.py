import re
import os
import json
import cv2
import shutil
from os.path import join
from tqdm import tqdm


def split_extension_name(name):
    extension_pattern = re.compile('\\.[a-zA-z]*$')
    search = extension_pattern.search(name)
    assert search is not None, 'wrong file! at name: {}'.format(name)
    search_start = search.regs[0][0]
    return name[:search_start], name[search_start:]


# train ================================================================================================================
from_path = '/simple_ssd/ys2/data_clear/20181112_merged'
to_path = '/simple_ssd/ys2/data_clear/human_detection_data/train'
to_path_image = join(to_path, 'image')
to_path_json = join(to_path, 'json')

os.makedirs(to_path_image, exist_ok=True)
os.makedirs(to_path_json, exist_ok=True)

for name in tqdm(os.listdir(from_path)):
    if split_extension_name(name)[1] == '.json':
        json_path = os.path.join(from_path, name)
        image_path = split_extension_name(json_path)[0]+'.jpg'

        new_json_path = join(to_path_json, name)
        new_image_path = join(to_path_image, name)
        new_image_path = split_extension_name(new_image_path)[0] + '.jpg'

        new_jsn = dict()

        im = cv2.imread(image_path)
        if im is None:
            continue
        im_height, im_width, im_channels = im.shape

        new_jsn['image_size'] = {'h': im_height, 'w': im_width, 'c': im_channels}

        jsn = json.load(open(json_path))
        if len(jsn['human_list']) == 0:
            continue
        new_jsn['image_name'] = jsn['human_list'][0][0]

        new_human_list = list()
        for human in jsn['human_list']:
            new_human = dict()

            xmin_flt = float(human[4])
            xmax_flt = float(human[5])
            ymin_flt = float(human[6])
            ymax_flt = float(human[7])

            xmin = int(xmin_flt*im_width)
            ymin = int(ymin_flt*im_height)
            xmax = int(xmax_flt*im_width)
            ymax = int(ymax_flt*im_height)
            cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            new_human['human_rect'] = {'xmin': round(xmin_flt, 6),
                                       'ymin': round(ymin_flt, 6),
                                       'xmax': round(xmax_flt, 6),
                                       'ymax': round(ymax_flt, 6)}
            new_human['is_crowd'] = int(human[10])

            new_human_list.append(new_human)
        new_jsn['human_list'] = new_human_list

        # cv2.imshow('', im)
        # cv2.waitKey()

        with open(new_json_path, 'x') as fp:
            json.dump(new_jsn, fp)

        shutil.copy(image_path, new_image_path)
