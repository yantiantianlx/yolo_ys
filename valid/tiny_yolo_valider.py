import torch
from torch import nn
from torch.utils.data import DataLoader
from gear_tool.model_utils import one_card_model

from data.dataset import Hand_Dataset
from model.loss import YOLOLoss
from model.bbox_format import Bbox_Format
from model.predict_process import model_out_to_model_predict


class Valider:
    def __init__(self, arg):
        super().__init__()
        self.arg = arg

        self.dataset = Hand_Dataset(arg, arg.data.dataset.valid.root, im_size=arg.model.net.im_size, mode='valid')
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        self.loss_func = YOLOLoss()

        self.bbox_format = list()
        self.bbox_format.append(Bbox_Format(arg.model.net.im_size, arg.model.net.feature_size[0], arg.model.flt_anchor,
                                            arg.model.net.num_class, arg.model.mask_iou_threshold))
        self.bbox_format.append(Bbox_Format(arg.model.net.im_size, arg.model.net.feature_size[1], arg.model.flt_anchor,
                                            arg.model.net.num_class, arg.model.mask_iou_threshold))

    def load_net_weight(self, net):
        net.load_state_dict(one_card_model(torch.load(self.arg.model.weight_path)))

    def valid(self, net):
        net.eval()
        valid_loss_list = list()
        for batch_idx, batch_data_dict in enumerate(self.dataloader):
            batch_image = batch_data_dict['image'].to(next(net.parameters()).device)
            batch_label = batch_data_dict['label'].to(next(net.parameters()).device)
            batch_image_path = batch_data_dict['image_path']

            with torch.no_grad():
                net_out = net.forward(batch_image)
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
            valid_loss_list.append(whole_loss)

        net.train()
        return sum(valid_loss_list) / len(valid_loss_list)


if __name__ == '__main__':
    pass

