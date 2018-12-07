import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
from gear_config.yolo_config.body_micro_yolo_aug0 import ARG
from gear_tool.vis import VIS
from gear_tool.gear_utils import *
from gear_tool.model_utils import one_card_model
from gear_tool.lr_decay import LrSeeker

from data.dataset import Hand_Dataset
from model.body_micro_yolo.body_micro_yolo_net import YoloV3_Micro
from model.loss import YOLOLoss
from model.bbox_format import Bbox_Format
from model.predict_process import model_out_to_model_predict


# train()===============================================================================================================
def train(arg: ARG, train_loader, valid_loader, net, loss_func, optimizer, lr_decay, epoch, vis: VIS=None):
    bbox_format = list()
    bbox_format.append(Bbox_Format(arg.model.net.im_size, arg.model.net.feature_size[0], arg.model.flt_anchor, arg.model.net.num_class, arg.model.mask_iou_threshold))
    bbox_format.append(Bbox_Format(arg.model.net.im_size, arg.model.net.feature_size[1], arg.model.flt_anchor, arg.model.net.num_class, arg.model.mask_iou_threshold))
    net.train()

    whole_loss_list = list()
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

        vis.line('log_lr', np.log10(lr))
        vis.line('whole_loss', y=whole_loss.item())
        whole_loss_list.append(whole_loss.item())

        print('to step '+str(batch_idx)+': ', whole_loss_list)

        # for layer_idx, (losses, batch_predict, batch_target) in enumerate(zip(losses_list, predict_list, target_list)):
        #
        #     loss, loss_dx, loss_dy, loss_w, loss_h, loss_confidence, loss_class = losses
        #     vis.line(str(layer_idx)+'/loss', y=sum(loss.tolist()))
        #     vis.line(str(layer_idx)+'/loss_dx', y=sum(loss_dx.tolist()))
        #     vis.line(str(layer_idx)+'/loss_dy', y=sum(loss_dy.tolist()))
        #     vis.line(str(layer_idx)+'/loss_w', y=sum(loss_w.tolist()))
        #     vis.line(str(layer_idx)+'/loss_h', y=sum(loss_h.tolist()))
        #     vis.line(str(layer_idx)+'/loss_confidence', y=sum(loss_confidence.tolist()))
        #     # vis.line(str(layer_idx)+'/loss_class', y=sum(loss_class.tolist()))


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

    optimizer = torch.optim.Adam(net.parameters(), lr=arg.optim.lr, weight_decay=arg.optim.weight_decay)
    lr_decay = LrSeeker(1e-9, 1)

    for epoch in range(1, arg.train.epochs + 1):
        vis.epoch_counter = epoch-1
        train(arg, train_loader, valid_loader, net, loss_func, optimizer, lr_decay, epoch, vis)


if __name__ == '__main__':
    main()
