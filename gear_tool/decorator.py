
import torch
import numpy as np
from copy import copy
from collections import MutableSequence


def iterable_decorator(func):
    """
    make the func can handle sequence of type (MutableSequence, np.ndarray, torch.Tensor)
    :param func: one or more to one func
    :return:
    """
    def type_exchange_func(*args):
        if isinstance(args[0], (MutableSequence, np.ndarray, torch.Tensor)):
            res_as_input = copy(args[0])
            if len(args[0]) == 0:
                return args
            for idx, content in enumerate(zip(*args)):
                func_res = func(*content)
                if isinstance(func_res, float):
                    if isinstance(args[0], np.ndarray):
                        res_as_input = res_as_input.astype(np.float32)
                    if isinstance(args[0], torch.Tensor):
                        res_as_input = res_as_input.float()
                res_as_input[idx] = func_res
            return res_as_input

    return type_exchange_func


def rect_decorator(func):
    """
    decorate the point list, make it be bbox ???
    bbox should be x1y1x2y2
    :param func: one or more to one func
    :return:
    """
    def type_exchange_func(*args, **kwargs):
        bbox_list = list()
        if 'rect_list' in kwargs:
            bbox_list = kwargs['rect_list']
        if 'bbox_list' in kwargs:
            bbox_list = kwargs['bbox_list']
        points_list = list()
        for bbox in bbox_list:
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            points_list.append([x1, y1])
            points_list.append([x2, y2])



    return type_exchange_func
