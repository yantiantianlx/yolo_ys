import cv2
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from gear_tool.gear_utils import *
from gear_tool.model_utils import one_card_model
from gear_tool.conversion_utils import cxcywh_to_x1y1x2y2, torch_im_to_cv2_im

from data.dataset import Hand_Dataset
from model.bbox_format import Bbox_Format
from model.predict_process import model_out_to_model_predict, none_max_suppression, confidence_filter
from tst.map import det_eval


def bbox_iou_x1y1x2y2(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    # Intersection area
    inter_area = max(inter_rect_x2 - inter_rect_x1 + 1, 0) * \
                 max(inter_rect_y2 - inter_rect_y1 + 1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


class Tester:
    def __init__(self, arg):
        super().__init__()
        self.arg = arg

        self.dataset = Hand_Dataset(arg, arg.data.dataset.test.root, im_size=arg.model.net.im_size, mode='test')
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        self.bbox_format = list()
        self.bbox_format.append(Bbox_Format(arg.model.net.im_size, arg.model.net.feature_size[0], arg.model.flt_anchor,
                                            arg.model.net.num_class, arg.model.mask_iou_threshold))
        self.bbox_format.append(Bbox_Format(arg.model.net.im_size, arg.model.net.feature_size[1], arg.model.flt_anchor,
                                            arg.model.net.num_class, arg.model.mask_iou_threshold))

    def load_net_weight(self, net):
        net.load_state_dict(one_card_model(torch.load(self.arg.model.weight_path)))

    def tst(self, net, write_image=False):
        net.eval()
        target_dict = dict()
        predict_list = list()
        print('testing......')
        image_counter = 0
        for batch_idx, batch_data_dict in tqdm(enumerate(self.dataloader)):

            # # FOR TEST  FOR TEST  FOR TEST
            # if batch_idx > 10:
            #     break
            # # FOR TEST  FOR TEST  FOR TEST

            batch_image = batch_data_dict['image'].to(next(net.parameters()).device)
            batch_label = batch_data_dict['label'].to(next(net.parameters()).device)
            batch_image_path = batch_data_dict['image_path']

            label_list = list()
            for label_box in batch_label[0]:
                cls, xmid, ymid, w, h = label_box.tolist()
                if xmid + ymid + w + h > 0.000001:
                    im_w, im_h = self.arg.model.net.im_size
                    label_list.append([cls, xmid * im_w, ymid * im_h, w * im_w, h * im_h])

            target_dict[str(batch_idx) + '.jpg'] = label_list

            with torch.no_grad():
                net_out = net.forward(batch_image)
            predict_bbox_list = list()
            confidence_list = list()
            for model_out_idx, feature in enumerate(net_out):
                batch_predict = model_out_to_model_predict(feature, num_anchors=self.arg.model.net.num_anchor)
                batch_predict_bbox, batch_confidence = self.bbox_format[model_out_idx].to_bbox(*batch_predict)

                predict_bbox_list.append(batch_predict_bbox[0])  # notice: batch size == 1
                confidence_list.append(batch_confidence[0])  # notice: batch size == 1

            predict_bboxes = torch.cat(predict_bbox_list, 0)
            predict_bboxes = cxcywh_to_x1y1x2y2(predict_bboxes)
            confidences = torch.cat(confidence_list, 0)

            bboxes, confidences = confidence_filter(predict_bboxes, confidences,
                                                    self.arg.model.out_confidence_filter_threshold)
            bboxes, confidences = none_max_suppression(bboxes, confidences, self.arg.model.nms_iou_threshold)

            for box, conf in zip(bboxes, confidences):
                xmin, ymin, xmax, ymax = box.tolist()
                conf = conf.item()
                predict_list.append(
                    [str(batch_idx) + '.jpg', 0, conf, (xmin + xmax) // 2, (ymin + ymax) // 2, xmax - xmin,
                     ymax - ymin])

            if write_image:
                np_im = torch_im_to_cv2_im(batch_image[0]).copy()
                np_im = cv2.cvtColor(np_im, cv2.COLOR_RGB2BGR)
                np_im = cv2.resize(np_im, (448, 448))

                for label_box in batch_label[0]:
                    cls, xmid, ymid, w, h = label_box.tolist()
                    if xmid + ymid + w + h > 0.000001:
                        im_h, im_w, _ = np_im.shape
                        label_list.append([cls, xmid * im_w, ymid * im_h, w * im_w, h * im_h])
                        xmin, ymin, xmax, ymax = cxcywh_to_x1y1x2y2(xmid, ymid, w, h)
                        gt_xmin, gt_ymin, gt_xmax, gt_ymax = int(xmin * im_w), int(ymin * im_h), int(xmax * im_w), int(ymax * im_h)
                        np_im = cv2.rectangle(np_im, (gt_xmin, gt_ymin), (gt_xmax, gt_ymax), (0, 255, 0))

                for box, conf in zip(bboxes, confidences):
                    xmin, ymin, xmax, ymax = box.tolist()
                    conf = conf.item()
                    xmin = round(xmin*2)
                    xmax = round(xmax*2)
                    ymin = round(ymin*2)
                    ymax = round(ymax*2)
                    np_im = cv2.rectangle(np_im, (xmin, ymin), (xmax, ymax), (255, 255, 0))
                    np_im = cv2.putText(np_im, str(round(conf, 3)), ((xmin + xmax) // 2, (ymin + ymax) // 2),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

                    iou = bbox_iou_x1y1x2y2([gt_xmin, gt_ymin, gt_xmax, gt_ymax], [xmin, ymin, xmax, ymax])
                    np_im = cv2.putText(np_im, str(round(iou, 3)), ((xmin + xmax) // 2, (ymin + ymax) // 2 + 20),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)

                # cv2.imshow('', np_im)
                # cv2.waitKey()

                os.makedirs('image_out', exist_ok=True)
                cv2.imwrite('image_out/' + str(image_counter) + '.jpg', np_im)
                image_counter += 1

        rec, prec, ap = det_eval(predict_list, target_dict, 0, 0.75)
        net.train()
        return ap


if __name__ == '__main__':
    from gear_config.yolo_config.body_micro_yolo_aug1 import ARG
    from model.body_micro_yolo.body_micro_yolo_net import YoloV3_Micro

    arg = ARG()

    net = YoloV3_Micro(class_num=arg.model.net.num_class, anchors_num=arg.model.net.num_anchor)
    net.load_state_dict(one_card_model(torch.load(arg.model.weight_path)))

    tster = Tester(arg)
    ap = tster.tst(net, True)
    print('ap=', ap)
