import math
from gear_tool.conversion_utils import *
from .model_utils import bbox_iou_x1y1x2y2


class Bbox_Format:
    def __init__(self, im_size, feature_size, flt_anchors_wh, num_class, mask_iou_threshold):
        im_w, im_h = im_size
        self.f_w, self.f_h = feature_size
        self.anchors_wh = [(a_w*im_w, a_h*im_h) for a_w, a_h in flt_anchors_wh]
        self.num_class = num_class
        self.mask_iou_threshold = mask_iou_threshold

        self.num_anchor = len(flt_anchors_wh)
        self.stride_w, self.stride_h = im_w / self.f_w, im_h / self.f_h
        self.scaled_anchors_wh = [(a_w / self.stride_w, a_h / self.stride_h) for a_w, a_h in self.anchors_wh]

    def to_model(self, bbox_target_flt):
        bs = bbox_target_flt.shape[0]
        device = bbox_target_flt.device

        mask = torch.zeros(bs, self.num_anchor, self.f_h, self.f_w, requires_grad=False).to(device)
        negative_mask = torch.ones(bs, self.num_anchor, self.f_h, self.f_w, requires_grad=False).to(device)
        target_dx = torch.zeros(bs, self.num_anchor, self.f_h, self.f_w, requires_grad=False).to(device)
        target_dy = torch.zeros(bs, self.num_anchor, self.f_h, self.f_w, requires_grad=False).to(device)
        target_w = torch.zeros(bs, self.num_anchor, self.f_h, self.f_w, requires_grad=False).to(device)
        target_h = torch.zeros(bs, self.num_anchor, self.f_h, self.f_w, requires_grad=False).to(device)
        target_confidence = torch.zeros(bs, self.num_anchor, self.f_h, self.f_w, requires_grad=False).to(device)
        if self.num_class != 0:
            target_class = torch.zeros(bs, self.num_anchor, self.f_h, self.f_w, self.num_class, requires_grad=False).to(device)
        else:
            target_class = torch.zeros(0, requires_grad=False).to(device)

        for batch_idx in range(bs):
            for bbox_idx in range(bbox_target_flt.shape[1]):
                if bbox_target_flt[batch_idx, bbox_idx].sum() == 0:
                    continue
                # Convert xywh to box, notice: x, y, w, h is in [0,1]
                gt_cx = bbox_target_flt[batch_idx, bbox_idx, 1] * self.f_w
                gt_cy = bbox_target_flt[batch_idx, bbox_idx, 2] * self.f_h
                gt_w = bbox_target_flt[batch_idx, bbox_idx, 3] * self.f_w
                gt_h = bbox_target_flt[batch_idx, bbox_idx, 4] * self.f_h
                # Get grid box indices
                gt_cx_floor = min(int(gt_cx), self.f_w-1)
                gt_cy_floor = min(int(gt_cy), self.f_h-1)
                gt_cx_floor = max(gt_cx_floor, 0)
                gt_cy_floor = max(gt_cy_floor, 0)
                # Get shape of gt box
                gt_box_rela_np = np.array([gt_cx - gt_cx_floor, gt_cy - gt_cy_floor, gt_w, gt_h])[np.newaxis, :]
                gt_box_rela_np = cxcywh_to_x1y1x2y2(gt_box_rela_np)
                gt_box_rela = torch.FloatTensor(gt_box_rela_np)
                # Get shape of anchor box
                anchor_rect_np = np.concatenate((np.ones((self.num_anchor, 2))/2, np.array(self.scaled_anchors_wh)), 1)
                anchor_rect_np = cxcywh_to_x1y1x2y2(anchor_rect_np)
                anchor_rect = torch.FloatTensor(anchor_rect_np)
                # Calculate iou between gt and anchor shapes
                anchor_ious = bbox_iou_x1y1x2y2(gt_box_rela, anchor_rect)
                # Where the overlap is larger than threshold set mask to zero (ignore)
                negative_mask[batch_idx, anchor_ious > self.mask_iou_threshold, gt_cy_floor, gt_cx_floor] = 0
                # Find the best matching anchor box
                best_n = np.argmax(anchor_ious)
                # Masks
                mask[batch_idx, best_n, gt_cy_floor, gt_cx_floor] = 1
                # Coordinates
                target_dx[batch_idx, best_n, gt_cy_floor, gt_cx_floor] = gt_cx - gt_cx_floor
                target_dy[batch_idx, best_n, gt_cy_floor, gt_cx_floor] = gt_cy - gt_cy_floor
                # Width and height
                target_w[batch_idx, best_n, gt_cy_floor, gt_cx_floor] = math.log(gt_w / self.scaled_anchors_wh[best_n][0] + 1e-16)
                target_h[batch_idx, best_n, gt_cy_floor, gt_cx_floor] = math.log(gt_h / self.scaled_anchors_wh[best_n][1] + 1e-16)
                # object
                target_confidence[batch_idx, best_n, gt_cy_floor, gt_cx_floor] = 1
                # One-hot encoding of label
                if self.num_class != 0:
                    target_class[batch_idx, best_n, gt_cy_floor, gt_cx_floor, int(bbox_target_flt[batch_idx, bbox_idx, 0])] = 1

        return [mask, negative_mask, target_dx, target_dy, target_w, target_h, target_confidence, target_class]

    def to_bbox(self, predict_dx, predict_dy, predict_w, predict_h, predict_confidence, predict_class):
        bs = predict_dx.size(0)
        device = predict_dx.device

        # Calculate offsets for each grid
        grid_x = torch.linspace(0, self.f_w - 1, self.f_w).repeat(self.f_w, 1).repeat(
            bs * self.num_anchor, 1, 1).view(predict_dx.shape).type(torch.FloatTensor).to(device)
        grid_y = torch.linspace(0, self.f_h - 1, self.f_h).repeat(self.f_h, 1).t().repeat(
            bs * self.num_anchor, 1, 1).view(predict_dy.shape).type(torch.FloatTensor).to(device)

        # Calculate anchor w, h
        anchor_w = torch.FloatTensor(self.scaled_anchors_wh).index_select(1, torch.LongTensor([0]))
        anchor_h = torch.FloatTensor(self.scaled_anchors_wh).index_select(1, torch.LongTensor([1]))
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, self.f_h * self.f_w).view(predict_w.shape).to(device)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, self.f_h * self.f_w).view(predict_h.shape).to(device)

        # Add offset and scale with anchors
        pred_boxes = torch.FloatTensor(bs, self.num_anchor, self.f_h, self.f_w, 4).to(device)
        pred_boxes[..., 0] = predict_dx.data + grid_x
        pred_boxes[..., 1] = predict_dy.data + grid_y
        pred_boxes[..., 2] = torch.exp(predict_w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(predict_h.data) * anchor_h

        _scale = torch.FloatTensor([self.stride_w, self.stride_h] * 2).to(device)
        pred_boxes = pred_boxes * _scale

        # pred_boxes = pred_boxes * (predict_confidence > self.filter_iou_threshold).unsqueeze(-1).type(FloatTensor)

        return pred_boxes.view(bs, -1, 4), predict_confidence.view(bs, -1)
