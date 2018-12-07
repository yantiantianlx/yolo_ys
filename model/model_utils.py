import torch
import numpy as np


def bbox_iou_x1y1x2y2(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    assert len(box1) > 0 and len(box2) > 0
    assert type(box1) == type(box2)
    assert isinstance(box1, (list, np.ndarray, torch.Tensor))
    assert isinstance(box2, (list, np.ndarray, torch.Tensor))

    if isinstance(box1, torch.Tensor):
        max_func = torch.max
        min_func = torch.min
        expand_func = torch.unsqueeze
    else:
        max_func = np.maximum
        min_func = np.minimum
        expand_func = np.expand_dims

    list_flag = False
    if isinstance(box1, list):
        list_flag = True
        box1 = np.array(box1)
        box2 = np.array(box2)

    if len(box1.shape) == 1 and len(box2.shape) == 1:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    else:
        if len(box1.shape) == 1 and len(box2.shape) == 2:
            box1 = expand_func(box1, 0)

        if len(box1.shape) == 2 and len(box2.shape) == 1:
            box2 = expand_func(box2, 0)

        assert len(box1.shape) == len(box2.shape) == 2

        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = max_func(b1_x1, b2_x1)
    inter_rect_y1 = max_func(b1_y1, b2_y1)
    inter_rect_x2 = min_func(b1_x2, b2_x2)
    inter_rect_y2 = min_func(b1_y2, b2_y2)
    # Intersection area
    inter_area = max_func(inter_rect_x2 - inter_rect_x1 + 1, 0) * max_func(inter_rect_y2 - inter_rect_y1 + 1, 0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    union_area = b1_area + b2_area - inter_area

    iou = inter_area / (union_area + 1e-16)

    return iou.tolist() if list_flag else iou


if __name__ == '__main__':
    box1 = [0, 0, 200, 200]
    box2 = [100, 100, 300, 300]

    a = bbox_iou_x1y1x2y2(box1, box2)
    print(a)