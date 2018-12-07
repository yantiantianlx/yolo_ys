
import numpy as np
# import intervals as I
# pip install python-intervals


def filter_bbox(human_list, min_w_flt, min_h_flt):
    cleared_human_list = list()
    for human in human_list:
        x1, y1, x2, y2 = human
        w = x2 - x1
        h = y2 - y1
        # if w < min_w_flt:
        #     continue
        # if h < min_h_flt:
        #     continue
        if w*h < 0.01:
            continue
        cleared_human_list.append(human)
    # assert len(cleared_human_list) > 0
    return cleared_human_list


def x1y1x2y2_to_cxcywh(*args):
    if len(args) == 1:
        box = args[0]
        assert isinstance(args[0], list)
        if len(box.shape) == 1:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            w = box[2] - box[0]
            h = box[3] - box[1]
            box[0], box[1], box[2], box[3] = cx, cy, w, h
            return box.tolist()
        if len(box.shape) == 2:
            cx = (box[:, 0] + box[:, 2]) / 2
            cy = (box[:, 1] + box[:, 3]) / 2
            w = box[:, 2] - box[:, 0]
            h = box[:, 3] - box[:, 1]
            box[:, 0], box[:, 1], box[:, 2], box[:, 3] = cx, cy, w, h
            return box.tolist()
        raise Exception('check here, you get no output')


def cxcywh_to_x1y1x2y2(*args):
    if len(args) == 1:
        box = args[0]
        assert isinstance(args[0], list)
        if len(box.shape) == 1:
            x1 = box[0] - box[2] / 2
            x2 = box[0] + box[2] / 2
            y1 = box[1] - box[3] / 2
            y2 = box[1] + box[3] / 2
            box[0], box[1], box[2], box[3] = x1, y1, x2, y2
            return box.tolist()
        if len(box.shape) == 2:
            x1 = box[:, 0] - box[:, 2] / 2
            x2 = box[:, 0] + box[:, 2] / 2
            y1 = box[:, 1] - box[:, 3] / 2
            y2 = box[:, 1] + box[:, 3] / 2
            box[:, 0], box[:, 1], box[:, 2], box[:, 3] = x1, y1, x2, y2
            return box.tolist()
        raise Exception('check here, you get no output')


def scale_coordinate(coord_list, scale):
    new_coord_list = list()
    for coord in coord_list:
        x1, y1, x2, y2 = coord
        new_x1 = (x1-0.5)*scale + 0.5
        new_y1 = (y1-0.5)*scale + 0.5
        new_x2 = (x2-0.5)*scale + 0.5
        new_y2 = (y2-0.5)*scale + 0.5

        new_coord_list.append([new_x1, new_y1, new_x2, new_y2])
    return new_coord_list


# def get_coordinate_gap(coord_list):
#     x_interval = x_interval.union(I.closed(x1, x2))
#     y_interval = I.empty()
#     for coord in coord_list:
#         x1, y1, x2, y2 = coord
#         x_interval = x_interval.union(I.closed(x1, x2))
#         y_interval = y_interval.union(I.closed(y1, y2))
#     x_min_inerval = x_interval.enclosure()
#     y_min_inerval = y_interval.enclosure()
#     x_gap = x_min_inerval - x_interval
#     y_gap = y_min_inerval - y_interval
#     x_gap_list = list(x_gap)
#     y_gap_list = list(y_gap)
#     xn = np.random.randint(0, len(x_gap_list)-1)
#     x_gap_select = x_gap_list[xn]
#     x
#     yn = np.random.randint(0, len(y_gap_list)-1)
#     y_gap_select = y_gap_list[yn]
#     return


def get_scale_dx_dy(human_rect_list, min_w_flt, min_h_flt, max_scale=1.0):
    human_rect_list = filter_bbox(human_rect_list, min_w_flt, min_h_flt)
    max_w, max_h = 0., 0.
    max_w_idx, max_h_idx = 0, 0
    big_rect = [1., 1., 0., 0.]  # x1, y1, x2, y2
    for idx, human_rect in enumerate(human_rect_list):
        x1, y1, x2, y2 = human_rect
        big_rect[0] = min(big_rect[0], x1)
        big_rect[1] = min(big_rect[1], y1)
        big_rect[2] = max(big_rect[2], x2)
        big_rect[3] = max(big_rect[3], y2)

        w, h = x2-x1, y2-y1
        if w > max_w:
            max_w = w
            max_w_idx = idx
        if h > max_h:
            max_h = h
            max_h_idx = idx
    max_scale = min(1/(big_rect[2]-big_rect[0]), 1/(big_rect[3]-big_rect[1]))

    w__ = max(human_rect_list[max_w_idx][2]-human_rect_list[max_w_idx][0], human_rect_list[max_h_idx][2]-human_rect_list[max_h_idx][0])
    h__ = max(human_rect_list[max_w_idx][3]-human_rect_list[max_w_idx][1], human_rect_list[max_h_idx][3]-human_rect_list[max_h_idx][1])

    min_scale = max(min_w_flt/w__, min_h_flt/h__)
    scale = np.random.uniform(low=min_scale, high=max_scale)

    scaled_human_rect_list = scale_coordinate(human_rect_list, scale)

    scaled_human_rect_list = filter_bbox(scaled_human_rect_list, min_w_flt, min_h_flt)
    big_rect_scale = [1., 1., 0., 0.]  # x1, y1, x2, y2
    for idx, human_rect in enumerate(scaled_human_rect_list):
        x1, y1, x2, y2 = human_rect
        big_rect_scale[0] = min(big_rect_scale[0], x1)
        big_rect_scale[1] = min(big_rect_scale[1], y1)
        big_rect_scale[2] = max(big_rect_scale[2], x2)
        big_rect_scale[3] = max(big_rect_scale[3], y2)

    if scale <= 1.0 or True:  # TODO
        dx = np.random.uniform(0.-big_rect_scale[0], 1-big_rect_scale[2])
        dy = np.random.uniform(0.-big_rect_scale[1], 1-big_rect_scale[3])
    else:
        pass  # TODO

    return scale, dx, dy


if __name__ == '__main__':
    coord_list = [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]]
    get_scale_dx_dy(coord_list, 0, 0, 1)
