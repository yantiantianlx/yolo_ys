import numpy as np
import torch


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


def calculate_ap(rec, prec):

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def calculate_recall_and_precision(gt_bbox,
                                   predict_bbox_list,
                                   confidence,
                                   iou_threshold,
                                   ):
    assert len(gt_bbox) > 0
    gt_occupy = np.zeros(len(gt_bbox))

    sorted_ind = np.argsort(-confidence)
    predict_bbox_list = predict_bbox_list[sorted_ind, :]

    predict_len = len(predict_bbox_list)
    true_positive = np.zeros(predict_len)
    false_positive = np.zeros(predict_len)

    for pre_idx in range(predict_len):
        predict_bbox = predict_bbox_list[pre_idx]

        ious = bbox_iou_x1y1x2y2(predict_bbox, gt_bbox)
        iou_max = np.max(ious)
        iou_max_idx = np.argmax(ious)

        if iou_max > iou_threshold:
            if gt_occupy[iou_max_idx] == 0:
                true_positive[iou_max_idx] = 1
                gt_occupy[iou_max_idx] = 1
            else:
                false_positive[iou_max_idx] = 1
        else:
            false_positive[iou_max_idx] = 1

    fp = np.cumsum(false_positive)
    tp = np.cumsum(true_positive)

    recall = tp / float(len(gt_bbox))
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    return recall, precision


def draw_plot(mrec, mprec, ap):
    import matplotlib.pyplot as plt
    if True:
        plt.plot(mrec, mprec, '-o')
        # add a new penultimate point to the list (mrec[-2], 0.0)
        # since the last line segment (and respective area) do not affect the AP value
        area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
        area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
        plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
        # set window title
        fig = plt.gcf()  # gcf - get current figure
        fig.canvas.set_window_title('PR plot')
        # set plot title
        plt.title('AP: ' + str(ap))
        # plt.suptitle('This is a somewhat long figure title', fontsize=16)
        # set axis titles
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # optional - set axes
        axes = plt.gca()  # gca - get current axes
        axes.set_xlim([0.0, 1.0])
        axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
        # Alternative option -> wait for button to be pressed
        # while not plt.waitforbuttonpress(): pass # wait for key display
        # Alternative option -> normal display
        plt.show()
        # save the plot
        # fig.savefig(results_files_path + "/classes/" + class_name + ".png")
        plt.cla()  # clear axes for next plot


def calculate_map():
    pass


if __name__ == '__main__':
    # targets are in cls, x, y, w, h
    target_ano = {'1.jpg': [[1, 100, 100, 100, 100], [1, 300, 300, 20, 20], [1, 400, 400, 100, 100]],
                  '2.jpg': [[1, 100, 100, 100, 100], [1, 300, 300, 20, 20], [1, 400, 400, 100, 100]],
                  '3.jpg': [[1, 100, 100, 100, 100], [1, 300, 300, 20, 20], [1, 400, 400, 100, 100]],
                  }

    # dets are in cls, confidence, x, y, w, h
    dets = [
        ['1.jpg', 1, 0.95, 100, 100, 90, 90],
        ['1.jpg', 1, 0.7, 300, 300, 21, 21],
        ['1.jpg', 1, 0.8, 400, 400, 99, 99],
        ['1.jpg', 1, 0.9, 10, 10, 10, 10],

        ['2.jpg', 1, 0.95, 100, 100, 90, 90],
        ['2.jpg', 1, 0.85, 300, 300, 2, 2],
        ['2.jpg', 1, 0.8, 400, 400, 99, 99],
        ['2.jpg', 1, 0.9, 10, 10, 10, 10],
        ['2.jpg', 1, 0.3, 10, 10, 10, 10],
        ['2.jpg', 1, 0.4, 10, 10, 10, 10],
        ['2.jpg', 1, 0.3, 10, 10, 10, 10],
        ['2.jpg', 1, 0.2, 10, 10, 10, 10],

        ['3.jpg', 1, 0.95, 100, 100, 90, 90],
        ['3.jpg', 1, 0.6, 300, 300, 21, 21],
        ['3.jpg', 1, 0.8, 400, 400, 99, 99],
        ['3.jpg', 1, 0.9, 10, 10, 10, 10],
        ['3.jpg', 1, 0.3, 10, 10, 10, 10],
        ['3.jpg', 1, 0.2, 10, 10, 10, 10],
    ]

    rec, prec, ap = det_eval(dets, target_ano, 1, 0.75, plot=True)
    print(rec)
    print(prec)
    print(ap)
