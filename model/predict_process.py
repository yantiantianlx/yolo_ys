import torch
from .model_utils import bbox_iou_x1y1x2y2


def model_out_to_model_predict(model_out, num_anchors):
    bs = model_out.size(0)
    f_w, f_h = model_out.size(2), model_out.size(3)
    model_out = model_out.view(bs, num_anchors, -1, f_h, f_w).permute(0, 1, 3, 4, 2).contiguous()

    predict_dx = torch.sigmoid(model_out[..., 0])  # delta center x
    predict_dy = torch.sigmoid(model_out[..., 1])  # delta center y
    predict_w = model_out[..., 2]  # Width
    predict_h = model_out[..., 3]  # Height
    predict_confidence = torch.sigmoid(model_out[..., 4])  # Confidence predict
    predict_class = torch.sigmoid(model_out[..., 5:])  # Class predict

    return [predict_dx, predict_dy, predict_w, predict_h, predict_confidence, predict_class]


def confidence_filter(predict_bboxes, confidences, filter_threshold):
    if sum(confidences > filter_threshold) == 0:
        return [], []
    predict_bboxes_ = predict_bboxes[confidences > filter_threshold].float()
    confidences_ = confidences[confidences > filter_threshold].float()
    return predict_bboxes_, confidences_


def none_max_suppression(predict_bboxes, confidences, nms_threshold=0.4):
    if len(confidences) <= 1:
        return predict_bboxes, confidences
    boxes_conf = torch.cat((predict_bboxes, confidences.unsqueeze(-1)), -1)
    boxes_conf_list = sorted(boxes_conf, key=lambda s: s[-1], reverse=True)
    boxes_conf = torch.stack(boxes_conf_list, 0)
    first_idx = 0
    while boxes_conf.shape[0] > first_idx+1:
        first_ = boxes_conf[first_idx].unsqueeze(0)
        others_ = boxes_conf[first_idx+1:]
        iou = bbox_iou_x1y1x2y2(first_, others_)
        fit_iou_idx = iou < nms_threshold
        mask_idx = torch.cat((torch.Tensor([1]*(first_idx+1)).type_as(fit_iou_idx), fit_iou_idx), 0)
        boxes_conf = boxes_conf[mask_idx]
        first_idx += 1

    boxes = boxes_conf[:, 0:4]
    confis = boxes_conf[:, 4]
    return boxes, confis

