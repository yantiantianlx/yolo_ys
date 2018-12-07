import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import sys
import json

sys.path.append(os.path.dirname(os.getcwd()))
from gear_tool.vis import VIS
from gear_tool.gear_utils import *
from gear_tool.model_utils import one_card_model
from gear_tool.conversion_utils import cxcywh_to_x1y1x2y2, torch_im_to_cv2_im
from gear_config.yolo_config.tiny_yolo_default import ARG
from data.coco_dataset import COCO_Dataset
from model.tiny_yolo.tiny_yolo_net import YoloV3_Tiny
from model.loss import YOLOLoss
from model.bbox_format import Bbox_Format
from model.predict_process import model_out_to_model_predict, none_max_suppression, confidence_filter
from tst.tiny_yolo_tester import Tester
from valid.tiny_yolo_valider import Valider


class Trainer(nn.Module):
    def __init__(self, arg=None, vis=None, id=None):
        super().__init__()
        self.arg = ARG() if arg is None else arg
        self.vis = VIS(save_dir=arg.save.tensorboard) if vis is None else vis
        self.id = 0 if id is None else id

        self.global_step = 0
        self.hyper_parameter = dict()
        self.hyper_parameter['lr'] = 10**-1.6

        self.dataset = COCO_Dataset(self.arg, self.arg.data.dataset.train.root, im_size=self.arg.model.net.im_size,
                                    relative_path_txt=self.arg.data.dataset.train.txt_path)
        self.dataloader = DataLoader(self.dataset, batch_size=self.arg.data.dataloader.train.batch_size,
                                     shuffle=self.arg.data.dataloader.train.shuffle,
                                     drop_last=self.arg.data.dataloader.train.drop_last)
        # print('train dataloader len = ', len(self.dataloader))

        self.net = YoloV3_Tiny(class_num=self.arg.model.net.num_class, anchors_num=self.arg.model.net.num_anchor)
        input_to_net = torch.rand(1, 3, self.arg.model.net.im_size[0], self.arg.model.net.im_size[1])
        # self.net.forward(input_to_net)

        self.vis.model(self.net, input_to_net)
        if self.arg.model.weight_path is not None:
            self.net.load_state_dict(one_card_model(torch.load(self.arg.model.weight_path)))
        self.net = torch.nn.DataParallel(self.net)

        self.loss_func = YOLOLoss()
        self.loss_func = torch.nn.DataParallel(self.loss_func)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hyper_parameter['lr'])
        # self.lr_decay = ExponentialDecay(0.1, 10000, 1e-4)

        self.bbox_format = list()
        self.bbox_format.append(Bbox_Format(self.arg.model.net.im_size, self.arg.model.net.feature_size[0], self.arg.model.flt_anchor,
                                            self.arg.model.net.num_class, self.arg.model.mask_iou_threshold))
        self.bbox_format.append(Bbox_Format(self.arg.model.net.im_size, self.arg.model.net.feature_size[1], self.arg.model.flt_anchor,
                                            self.arg.model.net.num_class, self.arg.model.mask_iou_threshold))

        self.tester = Tester(self.arg)
        self.valider = Valider(self.arg)

    def evaluate(self):
        score = self.tester.tst(self.net)
        return score

    def validate(self):
        valid_loss = self.valider.valid(self.net)
        return valid_loss

    def save_weight(self, save_path):  # save net_weight
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.net.state_dict(), save_path)

    def step_train(self, batch_data_dict):
        self.net.train()
        batch_image = batch_data_dict['image'].to(next(self.net.parameters()).device)
        batch_label = batch_data_dict['label'].to(next(self.net.parameters()).device)
        batch_image_path = batch_data_dict['image_path']

        self.optimizer.zero_grad()
        net_out = self.net.forward(batch_image)
        predict_list = list()
        target_list = list()
        losses_list = list()
        whole_loss = 0
        for model_out_idx, feature in enumerate(net_out):
            batch_predict = model_out_to_model_predict(feature, num_anchors=self.arg.model.net.num_anchor)
            predict_list.append(batch_predict)

            batch_target = self.bbox_format[model_out_idx].to_model(batch_label)
            target_list.append(batch_target)

            batch_predict = list(filter(lambda x: x.shape != torch.Size([0]), batch_predict))
            batch_target = list(filter(lambda x: x.shape != torch.Size([0]), batch_target))
            layer_loss = self.loss_func.forward(batch_predict, batch_target)
            losses_list.append(layer_loss)
            whole_loss += torch.mean(layer_loss[0]) / len(net_out)
        whole_loss.backward()
        self.optimizer.step()

        return net_out, whole_loss, losses_list, predict_list, target_list

    def step_vis(self, batch_data_dict, net_out, whole_loss, losses_list, predict_list, target_list):
        batch_image = batch_data_dict['image'].to(next(self.net.parameters()).device)
        batch_label = batch_data_dict['label'].to(next(self.net.parameters()).device)

        predict_bbox_list = list()
        confidence_list = list()
        for model_out_idx, feature in enumerate(net_out):
            batch_predict = model_out_to_model_predict(feature, num_anchors=self.arg.model.net.num_anchor)
            batch_predict_bbox, batch_confidence = self.bbox_format[model_out_idx].to_bbox(*batch_predict)

            predict_bbox_list.append(batch_predict_bbox[0])
            confidence_list.append(batch_confidence[0])

        predict_bboxes = torch.cat(predict_bbox_list, 0)
        predict_bboxes = cxcywh_to_x1y1x2y2(predict_bboxes)
        confidences = torch.cat(confidence_list, 0)

        bboxes, confidences = confidence_filter(predict_bboxes, confidences,
                                                self.arg.model.out_confidence_filter_threshold)

        np_im = torch_im_to_cv2_im(batch_image[0]).copy()
        for box in bboxes:
            x1, y1, x2, y2 = list(map(lambda x: int(x), box))
            np_im = cv2.rectangle(np_im, (x1, y1), (x2, y2), (255, 255, 0))
        self.vis.image('_0_predict_before_nms', np_im)

        bboxes, confidences = none_max_suppression(bboxes, confidences, self.arg.model.nms_iou_threshold)

        np_im = torch_im_to_cv2_im(batch_image[0]).copy()
        for box, conf in zip(bboxes, confidences):
            x1, y1, x2, y2 = list(map(lambda x: int(x), box.tolist()))
            np_im = cv2.rectangle(np_im, (x1, y1), (x2, y2), (0, 255, 255))
            np_im = cv2.putText(np_im, str(round(conf.item(), 3)), ((x1 + x2) // 2, (y1 + y2) // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        self.vis.image('_0_predict_after_nms', np_im)

        image_show = torch_im_to_cv2_im(batch_image[0]).copy()
        for hand_idx in range(batch_label[0].shape[0]):
            cx, cy, w, h = (batch_label[0][hand_idx].data.cpu().numpy()[1:])
            x1, y1, x2, y2 = cxcywh_to_x1y1x2y2(cx, cy, w, h)
            x1, y1, x2, y2 = list(map(lambda x: int(x * self.arg.model.net.im_size[0]), (x1, y1, x2, y2)))
            image_show = cv2.rectangle(image_show, (x1, y1), (x2, y2), (0, 255, 0))
        self.vis.image('_0_target', image_show)

        self.vis.line('0_train_loss', y=whole_loss.item())

        for layer_idx, (losses, batch_predict, batch_target) in enumerate(
                zip(losses_list, predict_list, target_list)):

            mask, negative_mask, tar_dx, tar_dy, tar_w, tar_h, tar_confidence, tar_class = batch_target
            pre_dx, pre_dy, pre_w, pre_h, pre_confidence, pre_class = batch_predict

            batch_show_idx = 0
            for anchor_idx in range(self.arg.model.net.num_anchor):
                self.vis.image('layer' + str(layer_idx) + '_anchor' + str(anchor_idx) + '_confidence/target',
                               tar_confidence[batch_show_idx, anchor_idx])
                self.vis.image('layer' + str(layer_idx) + '_anchor' + str(anchor_idx) + '_confidence/predict',
                               pre_confidence[batch_show_idx, anchor_idx])
                # self.vis.image('layer'+str(layer_idx)+'_anchor'+str(anchor_idx)+'_class/target', tar_class[batch_show_idx, anchor_idx, :, :, 0])
                # self.vis.image('layer'+str(layer_idx)+'_anchor'+str(anchor_idx)+'_class/predict', pre_class[batch_show_idx, anchor_idx])

            loss, loss_dx, loss_dy, loss_w, loss_h, loss_confidence, loss_class = losses
            self.vis.line('layer' + str(layer_idx) + '/loss', y=sum(loss.tolist()))
            self.vis.line('layer' + str(layer_idx) + '/loss_dx', y=sum(loss_dx.tolist()))
            self.vis.line('layer' + str(layer_idx) + '/loss_dy', y=sum(loss_dy.tolist()))
            self.vis.line('layer' + str(layer_idx) + '/loss_w', y=sum(loss_w.tolist()))
            self.vis.line('layer' + str(layer_idx) + '/loss_h', y=sum(loss_h.tolist()))
            self.vis.line('layer' + str(layer_idx) + '/loss_confidence', y=sum(loss_confidence.tolist()))
            # self.vis.line(str(layer_idx)+'/loss_class', y=sum(loss_class.tolist()))

    def train_net(self, num_step):
        begin_step = self.global_step
        last_step = self.global_step + num_step - 1
        for _ in range(num_step//len(self.dataloader)+1):
            for batch_idx, batch_data_dict in enumerate(self.dataloader):
                print('train_step{}-->{}: now:{}'.format(begin_step, last_step, self.global_step))

                net_out, whole_loss, losses_list, predict_list, target_list = self.step_train(batch_data_dict)

                if self.global_step % self.arg.train.vis_interval == 0 and self.global_step != 0:
                    self.step_vis(batch_data_dict, net_out, whole_loss, losses_list, predict_list, target_list)

                if self.global_step % self.arg.train.valid_interval == 0 and self.global_step != 0:
                    valid_loss = self.validate()
                    self.vis.line('1_valid_loss', y=valid_loss.item())

                if self.global_step % self.arg.train.save_interval == 0 and self.global_step != 0:
                    save_path = os.path.join(self.arg.save.model, 'body_yolo_iter{}_weight'.format(self.global_step))
                    self.save_weight(save_path=save_path)

                if self.global_step % self.arg.train.test_interval == 0 and self.global_step != 0:
                    score = self.evaluate()
                    self.vis.line('2_test_AP', y=score.item())

                if self.global_step == last_step:
                    self.global_step += 1
                    self.vis.global_step = self.global_step
                    break
                self.global_step += 1
                self.vis.global_step = self.global_step


if __name__ == '__main__':
    from gear_config.yolo_config.tiny_yolo_default import ARG

    arg = ARG()
    prepare_save_dirs(arg)

    vis = VIS(save_dir=arg.save.tensorboard, show_dir=arg.save.tensorboard)

    trainer = Trainer(arg, vis)
    trainer.to('cuda')
    for i in range(50):
        trainer.train_net(1000)

