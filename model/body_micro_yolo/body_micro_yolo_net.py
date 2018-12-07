import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_


class Conv_BN_Relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x


class YoloV3_Micro(nn.Module):
    def __init__(self, class_num, anchors_num):
        super().__init__()
        out_c = (5+class_num)*anchors_num  # output_channel

        self.conv_bn_relu_0 = Conv_BN_Relu(3, 16)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_bn_relu_2 = Conv_BN_Relu(16, 32)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_bn_relu_4 = Conv_BN_Relu(32, 64)
        self.maxpool_5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_bn_relu_6 = Conv_BN_Relu(64, 128)
        self.maxpool_7 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_bn_relu_8 = Conv_BN_Relu(128, 256)
        self.maxpool_9 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_bn_relu_10 = Conv_BN_Relu(256, 256)
        self.conv_bn_relu_11 = Conv_BN_Relu(256, 256, kernel_size=1, padding=0)
        self.conv_bn_relu_12 = Conv_BN_Relu(256, 256)
        self.conv_13 = nn.Conv2d(256, out_c, kernel_size=1, padding=0)

        self.conv_bn_relu_14 = Conv_BN_Relu(256, 128, kernel_size=1, padding=0)
        self.upsample_15 = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_bn_relu_17 = Conv_BN_Relu(128+256, 128)
        self.conv_18 = nn.Conv2d(128, out_c, kernel_size=1, padding=0)

        self.apply(init_weight)

    def forward(self, im):                # shape=(bs,3,320,320)
        x0 = self.conv_bn_relu_0(im)      # shape=(bs,16,320,320)
        x1 = self.maxpool_1(x0)           # shape=(bs,16,160,160)
        x2 = self.conv_bn_relu_2(x1)      # shape=(bs,32,160,160)
        x3 = self.maxpool_3(x2)           # shape=(bs,32,80,80)
        x4 = self.conv_bn_relu_4(x3)      # shape=(bs,64,80,80)
        x5 = self.maxpool_5(x4)           # shape=(bs,64,40,40)
        x6 = self.conv_bn_relu_6(x5)      # shape=(bs,128,40,40)
        x7 = self.maxpool_7(x6)           # shape=(bs,128,20,20)
        x8 = self.conv_bn_relu_8(x7)      # shape=(bs,256,20,20)

        x9 = self.maxpool_9(x8)           # shape=(bs,256,10,10)
        x10 = self.conv_bn_relu_10(x9)     # shape=(bs,512,10,10)
        x11 = self.conv_bn_relu_11(x10)    # shape=(bs,512,10,10)
        x12 = self.conv_bn_relu_12(x11)   # shape=(bs,256,10,10)
        x13 = self.conv_13(x12)           # shape=(bs,out_c,10,10)

        x14 = self.conv_bn_relu_14(x11)   # shape=(bs,128,10,10)
        x15 = self.upsample_15(x14)       # shape=(bs,128,20,20)

        x16 = torch.cat((x8, x15), 1)     # shape=(bs,384,20,20)
        x17 = self.conv_bn_relu_17(x16)   # shape=(bs,128,20,20)
        x18 = self.conv_18(x17)           # shape=(bs,out_c,20,20)

        return x13, x18


def init_weight(m):
    if isinstance(m, nn.Conv2d):
        xavier_uniform_(m.weight.data)


def get_input():
    return torch.rand(1, 3, 224, 224)


if __name__ == '__main__':
    # from gear_config.yolo_config.body_micro_yolo_V2_default import ARG
    # arg = ARG()

    net = YoloV3_Micro(class_num=1, anchors_num=3)
    input_to_net = torch.rand(1, 3, 224, 224)
    net.forward(input_to_net)
    torch.save(net.state_dict(), 'test_for_size_model')

