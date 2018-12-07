from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_


def one_card_model(state_dict):
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict


def force_load_pretrain(net: nn.Module, pre_train_model_path):
    pretrained_dict = torch.load(pre_train_model_path, map_location=lambda storage, loc: storage)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    pretrained_dict = one_card_model(pretrained_dict)

    notrain_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in notrain_dict}

    notrain_dict.update(pretrained_dict)
    net.load_state_dict(notrain_dict)
    return net


def init_weight(m):
    if isinstance(m, nn.Conv2d):
        xavier_uniform_(m.weight.data)
