import cv2
import torch
import numpy as np


def cxcywh_to_x1y1x2y2(*args):
    if len(args) == 4:
        cx, cy, w, h = args
        x1 = cx - w/2
        x2 = cx + w/2
        y1 = cy - h/2
        y2 = cy + h/2
        return x1, y1, x2, y2

    if len(args) == 1:
        box = args[0]
        assert isinstance(args[0], (list, np.ndarray, torch.Tensor))
        list_flag = False
        if isinstance(box, list):
            list_flag = True
            box = np.array(box)
        if len(box.shape) == 1:
            x1 = box[0] - box[2] / 2
            x2 = box[0] + box[2] / 2
            y1 = box[1] - box[3] / 2
            y2 = box[1] + box[3] / 2
            box[0], box[1], box[2], box[3] = x1, y1, x2, y2
            return box.tolist() if list_flag else box
        if len(box.shape) == 2:
            x1 = box[:, 0] - box[:, 2] / 2
            x2 = box[:, 0] + box[:, 2] / 2
            y1 = box[:, 1] - box[:, 3] / 2
            y2 = box[:, 1] + box[:, 3] / 2
            box[:, 0], box[:, 1], box[:, 2], box[:, 3] = x1, y1, x2, y2
            return box.tolist() if list_flag else box
        raise Exception('check here, you get no output')


def x1y1x2y2_to_cxcywh(*args):
    if len(args) == 4:
        x1, y1, x2, y2 = args
        cx = (x1 + x2)/2
        cy = (y1 + y2)/2
        w = x2 - x1
        h = y2 - y1
        return cx, cy, w, h

    if len(args) == 1:
        box = args[0]
        assert isinstance(args[0], (list, np.ndarray, torch.Tensor))
        list_flag = False
        if isinstance(box, list):
            list_flag = True
            box = np.array(box)
        if len(box.shape) == 1:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            w = box[2] - box[0]
            h = box[3] - box[1]
            box[0], box[1], box[2], box[3] = cx, cy, w, h
            return box.tolist() if list_flag else box
        if len(box.shape) == 2:
            cx = (box[:, 0] + box[:, 2]) / 2
            cy = (box[:, 1] + box[:, 3]) / 2
            w = box[:, 2] - box[:, 0]
            h = box[:, 3] - box[:, 1]
            box[:, 0], box[:, 1], box[:, 2], box[:, 3] = cx, cy, w, h
            return box.tolist() if list_flag else box
        raise Exception('check here, you get no output')


def cv2_im_to_torch_im(cv2_image: np.ndarray):
    cv2_image = cv2_image / 255.
    transposed_im = np.transpose(cv2_image, (2, 0, 1))
    torch_image = torch.from_numpy(transposed_im).float()
    return torch_image


def torch_im_to_cv2_im(torch_image: torch.Tensor):
    np_im = torch_image.data.cpu().numpy()
    np_im = (np_im * 255).astype(np.uint8)
    transposed_np_im = np.transpose(np_im, (1, 2, 0))
    return transposed_np_im


def square_padding(cv2_image: np.ndarray, point_list=None, bbox_list=None):
    point_float_flag = False
    im_h, im_w, _ = cv2_image.shape
    dim_diff = np.abs(im_h - im_w)
    pad1, pad2 = dim_diff // 2, dim_diff // 2
    pad_hwc = ((pad1, pad2), (0, 0), (0, 0)) if im_h <= im_w else ((0, 0), (pad1, pad2), (0, 0))  # pad to h / pad to w
    padded_im = np.pad(cv2_image, pad_hwc, 'constant', constant_values=128)

    padded_h = padded_im.shape[0]

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

    res_list = [padded_im, pad_hwc]
    if len(whole_point_list):
        new_whole_point_list = list()
        for point in whole_point_list:
            x, y = point

            if x <= 1. and y <= 1.:
                point_float_flag = True
                x = int(x * im_w)
                y = int(y * im_h)

            padded_x = pad_hwc[1][0] + x
            padded_y = pad_hwc[0][0] + y

            if point_float_flag:
                padded_x = padded_x/padded_h
                padded_y = padded_y/padded_h

            new_whole_point_list.append([padded_x, padded_y])

        new_point_list = new_whole_point_list[:point_list_len]
        if len(new_point_list):
            res_list.append(new_point_list)

        new_rect_point_list = new_whole_point_list[point_list_len:]
        if len(new_rect_point_list):
            new_rect_list = list()
            for i in range(len(new_rect_point_list)//2):
                even_idx = 2*i
                odd_idx = 2*i + 1
                x1, y1 = new_rect_point_list[even_idx]
                x2, y2 = new_rect_point_list[odd_idx]
                new_rect_list.append([x1, y1, x2, y2])
            res_list.append(new_rect_list)

    return res_list


if __name__ == '__main__':
    pass

