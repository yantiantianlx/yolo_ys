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


class YoloV3_Tiny(nn.Module):
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
        self.maxpool_9 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv_bn_relu_10 = Conv_BN_Relu(256, 512)
        self.padding_0101 = nn.ZeroPad2d((0, 1, 0, 1))
        self.maxpool_11 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv_bn_relu_12 = Conv_BN_Relu(512, 1024)
        self.conv_bn_relu_13 = Conv_BN_Relu(1024, 256, kernel_size=1, padding=0)
        self.conv_bn_relu_14 = Conv_BN_Relu(256, 512)
        self.conv_15 = nn.Conv2d(512, out_c, kernel_size=1, padding=0)

        self.conv_bn_relu_18 = Conv_BN_Relu(256, 128, kernel_size=1, padding=0)
        self.upsample_19 = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_bn_relu_21 = Conv_BN_Relu(256+128, 256)
        self.conv_22 = nn.Conv2d(256, out_c, kernel_size=1, padding=0)

        self.apply(init_weight)

    def forward(self, im):                # shape=(bs,3,416,416)
        x0 = self.conv_bn_relu_0(im)      # shape=(bs,16,416,416)
        x1 = self.maxpool_1(x0)           # shape=(bs,16,208,208)
        x2 = self.conv_bn_relu_2(x1)      # shape=(bs,32,208,208)
        x3 = self.maxpool_3(x2)           # shape=(bs,32,104,104)
        x4 = self.conv_bn_relu_4(x3)      # shape=(bs,64,104,104)
        x5 = self.maxpool_5(x4)           # shape=(bs,64,52,52)
        x6 = self.conv_bn_relu_6(x5)      # shape=(bs,128,52,52)
        x7 = self.maxpool_7(x6)           # shape=(bs,128,26,26)
        x8 = self.conv_bn_relu_8(x7)      # shape=(bs,256,26,26)

        x9 = self.maxpool_9(x8)           # shape=(bs,256,13,13)
        x10 = self.conv_bn_relu_10(x9)    # shape=(bs,512,13,13)
        x10 = self.padding_0101(x10)      # shape=(bs,512,14,14)
        x11 = self.maxpool_11(x10)        # shape=(bs,1024,13,13)
        x12 = self.conv_bn_relu_12(x11)   # shape=(bs,1024,13,13)
        x13 = self.conv_bn_relu_13(x12)   # shape=(bs,256,13,13)
        x14 = self.conv_bn_relu_14(x13)   # shape=(bs,512,13,13)
        x15 = self.conv_15(x14)           # shape=(bs,out_c,13,13)

        x18 = self.conv_bn_relu_18(x13)   # shape=(bs,128,13,13)
        x19 = self.upsample_19(x18)       # shape=(bs,128,26,26)

        x20 = torch.cat((x8, x19), 1)     # shape=(bs,384,26,26)
        x21 = self.conv_bn_relu_21(x20)   # shape=(bs,256,26,26)
        x22 = self.conv_22(x21)           # shape=(bs,out_c,26,26)

        return x15, x22


def init_weight(m):
    if isinstance(m, nn.Conv2d):
        xavier_uniform_(m.weight.data)
