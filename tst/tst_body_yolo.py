import cv2
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from gear_tool.gear_utils import *
from gear_tool.model_utils import one_card_model
from gear_tool.conversion_utils import cxcywh_to_x1y1x2y2, torch_im_to_cv2_im

from data.dataset import Hand_Dataset
from model.body_micro_yolo.body_micro_yolo_net import YoloV3_Micro
from model.bbox_format import Bbox_Format
from model.predict_process import model_out_to_model_predict, none_max_suppression, confidence_filter
from tst.map import det_eval


# test() ===============================================================================================================
def tst(arg, test_loader, net, write_image=False):
    bbox_format = list()
    bbox_format.append(Bbox_Format(arg.model.net.im_size, arg.model.net.feature_size[0], arg.model.flt_anchor, arg.model.net.num_class, arg.model.mask_iou_threshold))
    bbox_format.append(Bbox_Format(arg.model.net.im_size, arg.model.net.feature_size[1], arg.model.flt_anchor, arg.model.net.num_class, arg.model.mask_iou_threshold))

    net.eval()
    target_dict = dict()
    predict_list = list()
    print('testing......')
    image_counter = 0
    for batch_idx, batch_data_dict in tqdm(enumerate(test_loader)):
        # if batch_idx > 10:
        #     break
        batch_image = batch_data_dict['image']
        batch_label = batch_data_dict['label']
        batch_image_path = batch_data_dict['image_path']

        np_im = torch_im_to_cv2_im(batch_image[0]).copy()

        label_list = list()
        for label_box in batch_label[0]:
            cls, xmid, ymid, w, h = label_box.tolist()
            if xmid+ymid+w+h > 0.000001:
                im_w, im_h = arg.model.net.im_size
                label_list.append([cls, xmid*im_w, ymid*im_h, w*im_w, h*im_h])
                xmin, ymin, xmax, ymax = cxcywh_to_x1y1x2y2(xmid, ymid, w, h)
                xmin, ymin, xmax, ymax = int(xmin*im_w), int(ymin*im_h), int(xmax*im_w), int(ymax*im_h)
                np_im = cv2.rectangle(np_im, (xmin, ymin), (xmax, ymax), (0, 255, 0))
        target_dict[str(batch_idx)+'.jpg'] = label_list

        batch_image, batch_label = list(map(lambda x: x.to(arg.train.device), (batch_image, batch_label)))

        net_out = net.forward(batch_image)
        predict_bbox_list = list()
        confidence_list = list()
        for model_out_idx, feature in enumerate(net_out):

            batch_predict = model_out_to_model_predict(feature, num_anchors=arg.model.net.num_anchor)
            batch_predict_bbox, batch_confidence = bbox_format[model_out_idx].to_bbox(*batch_predict)

            predict_bbox_list.append(batch_predict_bbox[0])  # notice: batch size == 1
            confidence_list.append(batch_confidence[0])      # notice: batch size == 1

        predict_bboxes = torch.cat(predict_bbox_list, 0)
        predict_bboxes = cxcywh_to_x1y1x2y2(predict_bboxes)
        confidences = torch.cat(confidence_list, 0)

        bboxes, confidences = confidence_filter(predict_bboxes, confidences, arg.model.out_confidence_filter_threshold)
        bboxes, confidences = none_max_suppression(bboxes, confidences, arg.model.nms_iou_threshold)

        for box, conf in zip(bboxes, confidences):
            xmin, ymin, xmax, ymax = box.tolist()
            conf = conf.item()
            predict_list.append([str(batch_idx)+'.jpg', 0, conf, (xmin + xmax) // 2, (ymin + ymax) // 2, xmax-xmin, ymax-ymin])

        if write_image:
            for box, conf in zip(bboxes, confidences):
                xmin, ymin, xmax, ymax = box.tolist()
                conf = conf.item()
                xmin = round(xmin)
                xmax = round(xmax)
                ymin = round(ymin)
                ymax = round(ymax)
                np_im = cv2.rectangle(np_im, (xmin, ymin), (xmax, ymax), (255, 255, 0))
                np_im = cv2.putText(np_im, str(round(conf, 3)), ((xmin + xmax) // 2, (ymin + ymax) // 2),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow('', np_im)
            cv2.waitKey()

            os.makedirs('image_out', exist_ok=True)
            cv2.imwrite('image_out/'+str(image_counter)+'.jpg', np_im)
            image_counter += 1

    rec, prec, ap = det_eval(predict_list, target_dict, 0, 0.5)
    return ap


if __name__ == '__main__':
    from gear_config.yolo_config.body_micro_yolo_aug1 import ARG

    arg = ARG()

    test_dataset = Hand_Dataset(arg, arg.data.dataset.test.root, im_size=arg.model.net.im_size, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    net = YoloV3_Micro(class_num=arg.model.net.num_class, anchors_num=arg.model.net.num_anchor)

    net = net.to(arg.test.device)
    if arg.model.weight_path is not None:
        net.load_state_dict(one_card_model(torch.load(arg.model.weight_path)))

    tst(arg, test_loader, net, write_image=True)

