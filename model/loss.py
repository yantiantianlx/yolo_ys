import torch.nn as nn
import torch


class YOLOLoss(nn.Module):
    def __init__(self,
                 ignore_threshold=0.5,
                 lambda_xy=2.5,
                 lambda_wh=2.5,
                 lambda_conf=1.0,
                 lambda_cls=1.0,
                 # lambda_xy=0.,
                 # lambda_wh=0.,
                 # lambda_conf=1.0,
                 # lambda_cls=0.,
                 ):
        super(YOLOLoss, self).__init__()

        self.ignore_threshold = ignore_threshold
        self.lambda_xy = lambda_xy
        self.lambda_wh = lambda_wh
        self.lambda_conf = lambda_conf
        self.lambda_cls = lambda_cls

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, predicts, targets):
        if len(predicts) == 6:
            pre_dx, pre_dy, pre_w, pre_h, pre_confidence, pre_class = predicts
            mask, negative_mask, tar_dx, tar_dy, tar_w, tar_h, tar_confidence, tar_class = targets
        else:
            pre_dx, pre_dy, pre_w, pre_h, pre_confidence = predicts
            mask, negative_mask, tar_dx, tar_dy, tar_w, tar_h, tar_confidence = targets

        loss_dx = self.bce_loss(pre_dx * mask, tar_dx * mask)
        loss_dy = self.bce_loss(pre_dy * mask, tar_dy * mask)
        loss_w = self.mse_loss(pre_w * mask, tar_w * mask)
        loss_h = self.mse_loss(pre_h * mask, tar_h * mask)

        loss_confidence = self.bce_loss(pre_confidence * mask, mask) + \
                          0.1 * self.bce_loss(pre_confidence * negative_mask, negative_mask * 0.0)

        if len(predicts) == 6:
            loss_class = self.bce_loss(pre_class[mask == 1], tar_class[mask == 1])
        else:
            loss_class = torch.tensor(0.).to(pre_dx.device)

        loss = self.lambda_xy*(loss_dx + loss_dy) + self.lambda_wh*(loss_w+loss_h) + \
               self.lambda_conf*loss_confidence + self.lambda_cls * loss_class

        return loss, loss_dx, loss_dy, loss_w, loss_h, loss_confidence, loss_class


