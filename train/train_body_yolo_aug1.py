import cv2
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
from gear_config.yolo_config.body_micro_yolo_aug1 import ARG
from gear_tool.vis import VIS
from gear_tool.gear_utils import *
from gear_tool.model_utils import one_card_model
from gear_tool.lr_decay import ExponentialDecay
from gear_tool.conversion_utils import cxcywh_to_x1y1x2y2, torch_im_to_cv2_im

from data.dataset import Hand_Dataset
from model.body_micro_yolo.body_micro_yolo_net import YoloV3_Micro
from model.loss import YOLOLoss
from model.bbox_format import Bbox_Format
from model.predict_process import model_out_to_model_predict, none_max_suppression, confidence_filter
from tst.tst_body_yolo import tst


# train()===============================================================================================================
def train(arg: ARG, train_loader, valid_loader, net, loss_func, optimizer, lr_decay, epoch, vis: VIS=None):
    bbox_format = list()
    bbox_format.append(Bbox_Format(arg.model.net.im_size, arg.model.net.feature_size[0], arg.model.flt_anchor, arg.model.net.num_class, arg.model.mask_iou_threshold))
    bbox_format.append(Bbox_Format(arg.model.net.im_size, arg.model.net.feature_size[1], arg.model.flt_anchor, arg.model.net.num_class, arg.model.mask_iou_threshold))
    net.train()
    for batch_idx, batch_data_dict in enumerate(train_loader):
        vis.iteration_counter = batch_idx
        print('Epoch:{}, step/all: {}/{}'.format(epoch, batch_idx, len(train_loader)))

        batch_image = batch_data_dict['image']
        batch_label = batch_data_dict['label']
        batch_image_path = batch_data_dict['image_path']
        batch_image, batch_label = list(map(lambda x: x.to(arg.train.device), (batch_image, batch_label)))

        lr = lr_decay.get_lr(global_step=vis.step)
        optimizer.param_groups[0]['lr'] = lr
        optimizer.zero_grad()
        net_out = net.forward(batch_image)
        predict_list = list()
        target_list = list()
        losses_list = list()
        whole_loss = 0
        for model_out_idx, feature in enumerate(net_out):
            batch_predict = model_out_to_model_predict(feature, num_anchors=arg.model.net.num_anchor)
            predict_list.append(batch_predict)

            batch_target = bbox_format[model_out_idx].to_model(batch_label)
            target_list.append(batch_target)

            batch_predict = list(filter(lambda x: x.shape != torch.Size([0]), batch_predict))
            batch_target = list(filter(lambda x: x.shape != torch.Size([0]), batch_target))
            layer_loss = loss_func.forward(batch_predict, batch_target)
            losses_list.append(layer_loss)
            whole_loss += torch.mean(layer_loss[0])/len(net_out)
        whole_loss.backward()
        optimizer.step()

        if batch_idx % arg.train.log_iteration_interval == 0:
            vis.line('log10_lr', np.log10(lr))

            predict_bbox_list = list()
            confidence_list = list()
            for model_out_idx, feature in enumerate(net_out):
                batch_predict = model_out_to_model_predict(feature, num_anchors=arg.model.net.num_anchor)
                batch_predict_bbox, batch_confidence = bbox_format[model_out_idx].to_bbox(*batch_predict)

                predict_bbox_list.append(batch_predict_bbox[0])
                confidence_list.append(batch_confidence[0])

            predict_bboxes = torch.cat(predict_bbox_list, 0)
            predict_bboxes = cxcywh_to_x1y1x2y2(predict_bboxes)
            confidences = torch.cat(confidence_list, 0)

            bboxes, confidences = confidence_filter(predict_bboxes, confidences, arg.model.out_confidence_filter_threshold)

            np_im = torch_im_to_cv2_im(batch_image[0]).copy()
            for box in bboxes:
                x1, y1, x2, y2 = list(map(lambda x: int(x), box))
                np_im = cv2.rectangle(np_im, (x1, y1), (x2, y2), (255, 255, 0))
            vis.image('_0_predict_before_nms', np_im)

            bboxes, confidences = none_max_suppression(bboxes, confidences, arg.model.nms_iou_threshold)

            np_im = torch_im_to_cv2_im(batch_image[0]).copy()
            for box, conf in zip(bboxes, confidences):
                x1, y1, x2, y2 = list(map(lambda x: int(x), box.tolist()))
                np_im = cv2.rectangle(np_im, (x1, y1), (x2, y2), (0, 255, 255))
                np_im = cv2.putText(np_im, str(round(conf.item(), 3)), ((x1+x2)//2, (y1+y2)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            vis.image('_0_predict_after_nms', np_im)

            image_show = torch_im_to_cv2_im(batch_image[0]).copy()
            for hand_idx in range(batch_label[0].shape[0]):
                cx, cy, w, h = (batch_label[0][hand_idx].data.cpu().numpy()[1:])
                x1, y1, x2, y2 = cxcywh_to_x1y1x2y2(cx, cy, w, h)
                x1, y1, x2, y2 = list(map(lambda x: int(x*arg.model.net.im_size[0]), (x1, y1, x2, y2)))
                image_show = cv2.rectangle(image_show, (x1, y1), (x2, y2), (0, 255, 0))
            vis.image('_0_target', image_show)

            vis.line('_0_whole_loss', y=whole_loss.item())

            for layer_idx, (losses, batch_predict, batch_target) in enumerate(zip(losses_list, predict_list, target_list)):

                mask, negative_mask, tar_dx, tar_dy, tar_w, tar_h, tar_confidence, tar_class = batch_target
                pre_dx, pre_dy, pre_w, pre_h, pre_confidence, pre_class = batch_predict

                batch_show_idx = 0
                for anchor_idx in range(arg.model.net.num_anchor):
                    vis.image('layer'+str(layer_idx)+'_anchor'+str(anchor_idx)+'_confidence/target', tar_confidence[batch_show_idx, anchor_idx])
                    vis.image('layer'+str(layer_idx)+'_anchor'+str(anchor_idx)+'_confidence/predict', pre_confidence[batch_show_idx, anchor_idx])
                    # vis.image('layer'+str(layer_idx)+'_anchor'+str(anchor_idx)+'_class/target', tar_class[batch_show_idx, anchor_idx, :, :, 0])
                    # vis.image('layer'+str(layer_idx)+'_anchor'+str(anchor_idx)+'_class/predict', pre_class[batch_show_idx, anchor_idx])

                loss, loss_dx, loss_dy, loss_w, loss_h, loss_confidence, loss_class = losses
                vis.line(str(layer_idx)+'/loss', y=sum(loss.tolist()))
                vis.line(str(layer_idx)+'/loss_dx', y=sum(loss_dx.tolist()))
                vis.line(str(layer_idx)+'/loss_dy', y=sum(loss_dy.tolist()))
                vis.line(str(layer_idx)+'/loss_w', y=sum(loss_w.tolist()))
                vis.line(str(layer_idx)+'/loss_h', y=sum(loss_h.tolist()))
                vis.line(str(layer_idx)+'/loss_confidence', y=sum(loss_confidence.tolist()))
                # vis.line(str(layer_idx)+'/loss_class', y=sum(loss_class.tolist()))

        if batch_idx % arg.train.valid_iteration_interval == 1:
            valid(arg, valid_loader, net, loss_func, epoch, vis)
            net.train()


# valid() ==============================================================================================================
def valid(arg, valid_loader, net, loss_func, epoch, vis):
    bbox_format = list()
    bbox_format.append(Bbox_Format(arg.model.net.im_size, arg.model.net.feature_size[0], arg.model.flt_anchor, arg.model.net.num_class, arg.model.mask_iou_threshold))
    bbox_format.append(Bbox_Format(arg.model.net.im_size, arg.model.net.feature_size[1], arg.model.flt_anchor, arg.model.net.num_class, arg.model.mask_iou_threshold))

    net.eval()

    valid_loss_list = list()
    for batch_idx, batch_data_dict in enumerate(valid_loader):
        batch_image = batch_data_dict['image']
        batch_label = batch_data_dict['label']
        batch_image_path = batch_data_dict['image_path']

        batch_image, batch_label = list(map(lambda x: x.to(arg.train.device), (batch_image, batch_label)))

        net_out = net.forward(batch_image)
        predict_list = list()
        target_list = list()
        losses_list = list()

        whole_loss = 0
        for model_out_idx, feature in enumerate(net_out):
            batch_predict = model_out_to_model_predict(feature, num_anchors=arg.model.net.num_anchor)
            predict_list.append(batch_predict)

            batch_target = bbox_format[model_out_idx].to_model(batch_label)
            target_list.append(batch_target)

            batch_predict = list(filter(lambda x: x.shape != torch.Size([0]), batch_predict))
            batch_target = list(filter(lambda x: x.shape != torch.Size([0]), batch_target))
            layer_loss = loss_func.forward(batch_predict, batch_target)
            losses_list.append(layer_loss)
            whole_loss += torch.mean(layer_loss[0])/len(net_out)
        valid_loss_list.append(whole_loss)
    vis.line('valid/whole_loss:', sum(valid_loss_list)/len(valid_loss_list))


# main() ===============================================================================================================
def main():
    arg = ARG()

    prepare_save_dirs(arg)
    vis = VIS(save_dir=arg.save.tensorboard)

    # =======================================================
    train_dataset = Hand_Dataset(arg, arg.data.dataset.train.root, im_size=arg.model.net.im_size, relative_path_txt=arg.data.dataset.train.txt_path)
    train_loader = DataLoader(train_dataset, batch_size=arg.data.dataloader.train.batch_size, shuffle=arg.data.dataloader.train.shuffle, drop_last=arg.data.dataloader.train.drop_last)
    vis.iteration_per_epoch = len(train_loader)

    valid_dataset = Hand_Dataset(arg, arg.data.dataset.valid.root, im_size=arg.model.net.im_size, mode='valid')
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    test_dataset = Hand_Dataset(arg, arg.data.dataset.test.root, im_size=arg.model.net.im_size, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    net = YoloV3_Micro(class_num=arg.model.net.num_class, anchors_num=arg.model.net.num_anchor)
    input_to_net = torch.rand(1, 3, arg.model.net.im_size[0], arg.model.net.im_size[1])
    net.forward(input_to_net)

    vis.model(net, input_to_net)
    net = net.to(arg.train.device)
    if arg.model.weight_path is not None:
        net.load_state_dict(one_card_model(torch.load(arg.model.weight_path)))
    net = torch.nn.DataParallel(net)

    loss_func = YOLOLoss(arg.train.device)
    loss_func = loss_func.to(arg.train.device)
    loss_func = torch.nn.DataParallel(loss_func)

    optimizer = torch.optim.Adam(net.parameters(), lr=arg.optim.lr)
    lr_decay = ExponentialDecay(0.1, 10000, 1e-4)

    for epoch in range(1, arg.train.epochs + 1):
        vis.epoch_counter = epoch-1
        train(arg, train_loader,valid_loader, net, loss_func, optimizer, lr_decay, epoch, vis)

        if epoch % arg.train.save_model_epoch_interval == 0:
            if arg.save.model is not None:
                torch.save(net.state_dict(), os.path.join(arg.save.model, 'body_micro_yolo_epoch{}_weight'.format(epoch)))

        if epoch % arg.train.test_epoch_interval == 0:
            ap = tst(arg, test_loader, net)
            vis.line('test/ap', ap)


if __name__ == '__main__':
    main()
