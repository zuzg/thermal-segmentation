import numpy as np
import torch
import torch.nn as nn

# based on https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets 

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class aa_embedding(nn.Module):
    """
    Linear -> ReLU -> Linear -> Reshape
    https://ww2.mini.pw.edu.pl/pprai2024/papers/105.pdf -> Figure 1b
    """

    # TODO: verify
    def __init__(self, in_ch, img_size):
        super(aa_embedding, self).__init__()
        self.img_size = img_size // 8
        self.linear1 = nn.Linear(in_ch, 64)  # 1 x 225
        self.relu = nn.ReLU()  # 1 x 225
        self.linear2 = nn.Linear(64, 1)  # 1 x 510

    def forward(self, x):
        s = x.shape[0]
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = x.expand(s, 1, self.img_size, self.img_size)
        return x


class UNet(nn.Module):
    """
    UNet: https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=1, out_ch=1, img_size=512):
        super(UNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.aa_encoding = aa_embedding(in_ch, img_size)

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])  # in_ch, 64
        self.Conv2 = conv_block(filters[0], filters[1])  # 64, 128
        self.Conv3 = conv_block(filters[1], filters[2])  # 128, 256
        self.Conv4 = conv_block(filters[2], filters[3])  # 256, 512
        self.Conv5 = conv_block(filters[3], filters[4])  # 512, 1024

        self.Up5 = up_conv(filters[4], filters[3])  # 1024, 512
        self.Up_conv5 = conv_block(filters[4] + 2, filters[3])

        self.Up4 = up_conv(filters[3], filters[2])  # 512, 256
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])  # 256, 128
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])  # 128, 64
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(
            filters[0], out_ch, kernel_size=1, stride=1, padding=0
        )  # 64, out_ch

        # TODO: add sigmoid
        # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        # take the information about the altitude and angle
        altitude = x[:, 1, 0, 0, np.newaxis]
        angle = x[:, 2, 0, 0, np.newaxis]
        # print(altitude.shape, angle.shape)
        altitude = self.aa_encoding(altitude)
        angle = self.aa_encoding(angle)
        # print(altitude.shape, angle.shape)
        # print(altitude, angle)

        # take only the first channel
        x = x[:, np.newaxis, 0]

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        print(e4.shape, d5.shape)
        # fuse as learnable features in latent space
        d5 = torch.cat((e4, d5, altitude, angle), dim=1)
        print(d5.shape)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        # d1 = self.active(out)
        return out
