import os
import time
import numpy as np
import subprocess
import sys

import torch
from torch import nn
from visdom import Visdom
import torchvision.utils as vutils
from tensorboardX import SummaryWriter


class VIS:
    """Vis class .

        a visualization tool for training/valid/test deep learning model.

    line:
        draw lines used in loss, accuracy etc.
    img:
        show img
    weight:
        show weights of a model.
    gradient:
        show gradient of all parameter of a model.
    model:
        show structure of a model.
    """
    def __init__(self, save_dir=None, show_dir=None, port=None):
        self.global_step = 0
        self.save_dir = None
        self.show_dir = None
        if save_dir is not None:
            self.add_save_dir(save_dir)
        if show_dir is not None:
            self.add_show_dir(show_dir, port=port)

    def add_save_dir(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(save_dir)

    def add_show_dir(self, show_dir, port=None):
        self.show_dir = show_dir
        # cmd = os.environ['HOME'] + "'/anaconda3/bin/'tensorboard --logdir=" + tensorboard_save_dir
        if port is None:
            cmd = sys.executable[:-6] + "tensorboard --logdir=" + show_dir
        else:
            cmd = sys.executable[:-6] + "tensorboard --logdir=" + show_dir + "--port=" + str(port)
        self.tb_log_out_process = subprocess.Popen("exec " + cmd, stdout=subprocess.PIPE, shell=True)

    def __del__(self):
        if self.show_dir is not None:
            self.tb_log_out_process.kill()

    def line(self, name, y: float):
        assert self.save_dir is not None
        self.tb_writer.add_scalar(name, y, self.global_step)

    def image(self, name, image, normalize=False, scale_each=False):
        assert self.save_dir is not None
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if image.dim() == 4:
            image = vutils.make_grid(image, normalize=normalize, scale_each=scale_each)
        if image.shape[-1] == 3:
            if image.dim() == 4:
                image = image.permute(0, 3, 1, 2)
            elif image.dim() == 3:
                image = image.permute(2, 0, 1)
        self.tb_writer.add_image(name, image, global_step=self.global_step)

    def weight(self, model):
        assert self.save_dir is not None
        for k, v in model.state_dict().items():
            self.tb_writer.add_histogram('weight' + '/' + k, v, global_step=self.global_step)
            if v.grad is not None:
                print(v)

    def gradient(self, model):
        assert self.save_dir is not None
        for m in model.named_modules():
            k, v = m[0], m[1]
            if isinstance(v, nn.Conv2d):
                w_grad = v.weight.grad
                self.tb_writer.add_histogram('gradient' + '/' + k + '_w_grad', w_grad, global_step=self.global_step)
                if v.bias is not None:
                    b_grad = v.bias.grad
                    self.tb_writer.add_histogram('gradient' + '/' + k + '_b_grad', b_grad, global_step=self.global_step)
            elif isinstance(v, nn.BatchNorm2d):
                pass
            elif isinstance(v, nn.Linear):
                w_grad = v.weight.grad
                self.tb_writer.add_histogram('gradient' + '/' + k + '_w_grad', w_grad, global_step=self.global_step)
                if v.bias is not None:
                    b_grad = v.bias.grad
                    self.tb_writer.add_histogram('gradient' + '/' + k + '_b_grad', b_grad, global_step=self.global_step)

    def model(self, model, input_to_model):
        assert self.save_dir is not None
        self.tb_writer.add_graph(model, input_to_model=input_to_model)


if __name__ == '__main__':

    myvis = VIS(save_dir='logs')
    for i in range(1, 10):
        # myvis.text('links', 'sdfasdfasdddddddddddddddddd')
        myvis.line('loss/train', -9 * i, i)
        myvis.line('loss/tst', 3 * i, i)
        time.sleep(1)
        print(i)
