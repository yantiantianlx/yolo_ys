import cv2
import torch
from gear_config.yolo_config.tiny_yolo_default import ARG
from gear_tool.gear_utils import *
from gear_tool.model_utils import one_card_model
from gear_tool.conversion_utils import yolo_mode_cv2_im_to_torch_im, cxcywh_to_x1y1x2y2, torch_im_to_cv2_im

from model.tiny_yolo.tiny_yolo_net import YoloV3_Tiny
from model.bbox_format import Bbox_Format
from model.predict_process import model_out_to_model_predict, none_max_suppression, confidence_filter


# tst() ================================================================================================================
def tst():
    arg = ARG()

    net = YoloV3_Tiny(class_num=arg.model.net.num_class, anchors_num=arg.model.net.num_anchor)

    net = net.to(arg.train.device)
    if arg.model.weight_path is not None:
        net.load_state_dict(one_card_model(torch.load(arg.model.weight_path)))
    net = torch.nn.DataParallel(net)
    net.eval()

    bbox_format = list()
    bbox_format.append(Bbox_Format(arg.model.net.im_size, arg.model.net.feature_size[0], arg.model.anchor, arg.model.net.num_class, arg.model.mask_iou_threshold))
    bbox_format.append(Bbox_Format(arg.model.net.im_size, arg.model.net.feature_size[1], arg.model.anchor, arg.model.net.num_class, arg.model.mask_iou_threshold))

    for image_path in os.listdir(arg.test.image_dir):
        abs_image_path = os.path.join(arg.test.image_dir, image_path)
        image = cv2.imread(abs_image_path)
        assert image is not None
        torch_im, pad_hwc = yolo_mode_cv2_im_to_torch_im(image, arg.model.net.im_size)
        batch_image = torch_im.unsqueeze(0)  # make up as a batch
        with torch.no_grad():
            net_out = net.forward(batch_image)

        predict_bbox_list = list()
        confidence_list = list()
        for model_out_idx, feature in enumerate(net_out):

            batch_predict = model_out_to_model_predict(feature, num_anchors=len(arg.model.anchor))
            batch_predict_bbox, batch_confidence = bbox_format[model_out_idx].to_bbox(*batch_predict)

            predict_bbox_list.append(batch_predict_bbox[0])  # notice: batch size == 1
            confidence_list.append(batch_confidence[0])      # notice: batch size == 1

        predict_bboxes = torch.cat(predict_bbox_list, 0)
        predict_bboxes = cxcywh_to_x1y1x2y2(predict_bboxes)
        confidences = torch.cat(confidence_list, 0)

        bboxes, confidences = confidence_filter(predict_bboxes, confidences, arg.model.out_confidence_filter_threshold)
        bboxes, confidences = none_max_suppression(bboxes, confidences, arg.model.nms_iou_threshold)

        np_im = torch_im_to_cv2_im(torch_im).copy()

        for box in bboxes:
            x1, y1, x2, y2 = list(map(lambda x: int(x), box))
            image_pre_show = cv2.rectangle(np_im, (x1, y1), (x2, y2), (255, 255, 0))

            # test_out_dir = arg.test.out_dir
            # cv2.imwrite(test_out_dir+'/'+image_path, image_pre_show)
            cv2.imshow(image_path, image_pre_show)
            cv2.waitKey()


if __name__ == '__main__':
    tst()