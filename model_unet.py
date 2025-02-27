import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(32, ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(32, ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class resconv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(resconv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(32, ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(32, ch_out),
            nn.ReLU(inplace=True)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        residual = self.Conv_1x1(x)
        x = self.conv(x)

        return residual + x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(32, ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(32, ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(32, F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(32, F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class U_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, first_layer_numKernel=64):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=first_layer_numKernel)
        self.Conv2 = conv_block(ch_in=first_layer_numKernel, ch_out=2 * first_layer_numKernel)
        self.Conv3 = conv_block(ch_in=2 * first_layer_numKernel, ch_out=4 * first_layer_numKernel)
        self.Conv4 = conv_block(ch_in=4 * first_layer_numKernel, ch_out=8 * first_layer_numKernel)
        self.Conv5 = conv_block(ch_in=8 * first_layer_numKernel, ch_out=16 * first_layer_numKernel)

        self.Up5 = up_conv(ch_in=16 * first_layer_numKernel, ch_out=8 * first_layer_numKernel)
        self.Up_conv5 = conv_block(ch_in=16 * first_layer_numKernel, ch_out=8 * first_layer_numKernel)

        self.Up4 = up_conv(ch_in=8 * first_layer_numKernel, ch_out=4 * first_layer_numKernel)
        self.Up_conv4 = conv_block(ch_in=8 * first_layer_numKernel, ch_out=4 * first_layer_numKernel)

        self.Up3 = up_conv(ch_in=4 * first_layer_numKernel, ch_out=2 * first_layer_numKernel)
        self.Up_conv3 = conv_block(ch_in=4 * first_layer_numKernel, ch_out=2 * first_layer_numKernel)

        self.Up2 = up_conv(ch_in=2 * first_layer_numKernel, ch_out=first_layer_numKernel)
        self.Up_conv2 = conv_block(ch_in=2 * first_layer_numKernel, ch_out=first_layer_numKernel)

        self.Conv_1x1 = nn.Sequential(
            nn.Conv2d(first_layer_numKernel, output_ch, kernel_size=1, stride=1, padding=0), nn.Sigmoid()  # Use sigmoid activation for binary segmentation
        )
        # self.Conv_1x1 =  nn.Conv2d(first_layer_numKernel, output_ch, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1