import cv2
import torch
from torch.utils.data import DataLoader
from gear_config.yolo_config.micro_yolo_default import ARG
from gear_tool.vis import VIS
from gear_tool.gear_utils import *
from gear_tool.model_utils import one_card_model
from gear_tool.conversion_utils import cxcywh_to_x1y1x2y2, torch_im_to_cv2_im

from data.dataset import Hand_Dataset
from model.micro_yolo.micro_yolo_net import YoloV3_Micro
from model.loss import YOLOLoss
from model.bbox_format import Bbox_Format
from model.predict_process import model_out_to_model_predict


# train()===============================================================================================================
def train(arg: ARG, train_loader, net, loss_func, optimizer, epoch, vis: VIS=None):
    bbox_format = list()
    bbox_format.append(Bbox_Format(arg.model.net.im_size, arg.model.net.feature_size[0], arg.model.anchor, arg.model.net.num_class, arg.model.mask_iou_threshold))
    bbox_format.append(Bbox_Format(arg.model.net.im_size, arg.model.net.feature_size[1], arg.model.anchor, arg.model.net.num_class, arg.model.mask_iou_threshold, arg.model.filter_iou_threshold))
    net.train()
    for batch_idx, batch_data_dict in enumerate(train_loader):
        vis.iteration_counter = batch_idx
        print('Epoch:{}, step/all: {}/{}'.format(epoch, batch_idx, len(train_loader)))

        batch_image = batch_data_dict['image']
        batch_label = batch_data_dict['label']
        batch_image_path = batch_data_dict['image_path']

        batch_image, batch_label = list(map(lambda x: x.to(arg.train.device), (batch_image, batch_label)))

        optimizer.zero_grad()
        net_out = net.forward(batch_image)
        predict_list = list()
        target_list = list()
        losses_list = list()
        whole_loss = 0
        for model_out_idx, feature in enumerate(net_out):
            batch_predict = model_out_to_model_predict(feature, num_anchors=len(arg.model.anchor))
            predict_list.append(batch_predict)

            batch_target = bbox_format[model_out_idx].to_model(batch_label)
            target_list.append(batch_target)

            layer_loss = loss_func.forward(batch_predict, batch_target)
            losses_list.append(layer_loss)
            whole_loss += torch.mean(layer_loss[0])/len(net_out)
        whole_loss.backward()
        optimizer.step()

        if batch_idx % arg.train.log_iteration_interval == 0:

            # # # # # TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TES
            # batch_predict_bbox, batch_confidence = bbox_format[model_out_idx].to_bbox(*batch_predict)
            #
            # image_show = torch_im_to_cv2_im(batch_image[0]).copy()
            # for pre_idx in range(batch_predict_bbox.shape[1]):
            #     cx, cy, w, h = (batch_predict_bbox[0][pre_idx].data.cpu().numpy())
            #     if np.sum((cx, cy, w, h)) < 0.000001:
            #         continue
            #     x1, y1, x2, y2 = cxcywh_to_x1y1x2y2(cx, cy, w, h)
            #     x1, y1, x2, y2 = list(map(lambda x: int(x), (x1, y1, x2, y2)))
            #     image_pre_show = cv2.rectangle(image_show, (x1, y1), (x2, y2), (255, 255, 0))
            #     vis.image('_0_predict'+str(model_out_idx), image_pre_show)
            # # cv2.imshow('_0_predict'+str(model_out_idx), image_pre_show)
            # # cv2.waitKey()
            # # # # # TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST

            vis.line('_0_whole_loss', y=whole_loss.item())

            image_show = torch_im_to_cv2_im(batch_image[0]).copy()
            for hand_idx in range(batch_label[0].shape[0]):
                cx, cy, w, h = (batch_label[0][hand_idx].data.cpu().numpy()[1:])
                x1, y1, x2, y2 = cxcywh_to_x1y1x2y2(cx, cy, w, h)
                x1, y1, x2, y2 = list(map(lambda x: int(x*160), (x1, y1, x2, y2)))
                image_show = cv2.rectangle(image_show, (x1, y1), (x2, y2), (0, 255, 0))
            vis.image('_0_target', image_show)
            # cv2.imshow('image_show', image_show)
            # cv2.waitKey()

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
                vis.line(str(layer_idx)+'/loss_class', y=sum(loss_class.tolist()))


# valid() ==============================================================================================================
def valid(arg, valid_loader, model, epoch, rvis):
    model.eval()
    pass


# main() ===============================================================================================================
def main():
    arg = ARG()

    prepare_save_dirs(arg)
    vis = VIS(save_dir=arg.save.tensorboard)

    # =======================================================
    hand_dataset = Hand_Dataset(arg, arg.data.dataset.train.root, arg.data.dataset.train.relative_txt_path, arg.model.net.im_size)
    train_loader = DataLoader(hand_dataset, batch_size=16, shuffle=False)
    vis.iteration_per_epoch = len(train_loader)

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

    for epoch in range(1, arg.train.epochs + 1):
        vis.epoch_counter = epoch
        train(arg, train_loader, net, loss_func, optimizer, epoch, vis)

        # if epoch % arg.train.valid_interval == 0:
        #     valid(arg, valid_loader, model, epoch, vis)

        if epoch % arg.train.save_model_epoch_interval == 0:
            torch.save(net.state_dict(), os.path.join(arg.save.model, 'micro_yolo_epoch{}_weight'.format(epoch)))


if __name__ == '__main__':
    main()
