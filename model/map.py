import numpy as np


def xywh_to_xyxy(xywh):
    """Convert [x1 y1 w h] box format to [x1 y1 x2 y2] format."""
    if isinstance(xywh, (list, tuple)) or (isinstance(xywh, np.ndarray) and xywh.shape == (4,)):
        # Single box given as a list of coordinates
        assert len(xywh) == 4
        x1, y1 = xywh[0], xywh[1]
        x2 = x1 + np.maximum(0., xywh[2] - 1.)
        y2 = y1 + np.maximum(0., xywh[3] - 1.)
        if isinstance(xywh, np.ndarray):
            return np.array((x1, y1, x2, y2))
        else:
            return (x1, y1, x2, y2)
    elif isinstance(xywh, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        try:
            return np.hstack(
                (xywh[:, 0:2], xywh[:, 0:2] + np.maximum(0, xywh[:, 2:4] - 1))
            )
        except Exception as e:
            print(e)
            print(xywh)
    else:
        raise TypeError('Argument xywh must be a list, tuple, or numpy array.')


def cal_ap(rec, prec, use_11_point=False):
    if use_11_point:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
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


def det_eval(dets,
             anos,
             class_id,
             ovthresh=0.75,
             use_11_point=False,
             plot=False):
    # first step, get all target anos corresponding to current class
    class_recs = {}
    npos = 0
    for image_id, ano in anos.items():
        bbox = np.array([x[1:5] for x in ano if x[0] == class_id])
        det = [False] * len(bbox)
        npos += len(det)
        class_recs[image_id] = {'bbox': bbox,
                                'det': det}

    # second step, get detection results
    image_ids = []
    clss = []
    confidence = []
    BB = []
    for det in dets:
        if det[1] == class_id:
            image_ids.append(det[0])
            clss.append(det[1])
            confidence.append(det[2])
            BB.append(det[3:])
    confidence = np.array(confidence)
    BB = np.array(BB)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        bb = xywh_to_xyxy(bb)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        BBGT = xywh_to_xyxy(BBGT)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)  # what's the sue of npos?
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = cal_ap(rec, prec, use_11_point)

    if plot:
        draw_plot(rec, prec, ap)
    return rec, prec, ap


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
