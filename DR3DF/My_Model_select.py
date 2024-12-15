import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy
from My_Model_dehazing import *


class two_weight(nn.Module):
    def __init__(self):
        super(two_weight, self).__init__()
        self.F = nn.Flatten(3, 4)
        # self.avg_layer = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.avg_layer = nn.AdaptiveAvgPool2d(1)
        # 1x1 Conv layer
        # self.conv = nn.Conv2d(in_channels=s_size, out_channels=1, kernel_size=1)
        self.conv = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1)
        # Hard Sigmoid
        self.hard_sigmoid = nn.Hardsigmoid()
        self.soft = nn.Softmax(dim=1)
        # ReLU function
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x1 = torch.unsqueeze(x1, 1)
        x2 = torch.unsqueeze(x2, 1)
        x = self.F(torch.cat((x1, x2), 1))  # batch_size*L*C*S
        # x =x.transpose(3,1)   #batch_size*S*C*L
        x = self.avg_layer(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.soft(x)
        # pi_L = self.hard_sigmoid(x)
        temp = x[0, :, 0, 0]
        return temp


class TDP_Attention(nn.Module):
    # ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias,
    def __init__(self, channel, r=8, norm_layer='', act=nn.ReLU):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // r, 1, padding=0, bias=True),
            # norm_layer(channel // 8),
            act())
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel // r, 1, padding=0, bias=True),
            # norm_layer(channel // 8),
            act())

        self.fcs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(channel // r, channel, 1, padding=0, bias=True),
            # norm_layer(channel // 8),
        )
            for _ in range(2)])
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel // r, 1, 1, padding=0, bias=True),
                # norm_layer(channel // 8),
            )
            for _ in range(2)
        ])
        self.soft = nn.Softmax(dim=1)
        self.conv1 = nn.Conv2d(channel, channel, (1, 1))
        self.conv2 = nn.Conv2d(channel, channel, (1, 1))

    def forward(self, x1, x2, w1, w2):
        x = x1 * w1 + x2 * w2
        temp = self.pool(x)
        t1 = self.fc(temp)
        t2 = self.conv(x)
        arr1 = [fc(t1) for fc in self.fcs]
        arr2 = [conv(t2) for conv in self.convs]
        y1 = self.conv1(arr1[0] + arr2[0])
        y2 = self.conv2(arr1[1] + arr2[1])
        w1 = (y1).unsqueeze_(dim=1)
        w2 = (y2).unsqueeze_(dim=1)

        w = self.soft(torch.cat([w1, w2], dim=1))
        # out =  x1 * w[:, 0, ::] + x2 * w[:, 1, ::]
        return x1 * w[:, 0, ::] + x2 * w[:, 1, ::]

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class ScoreBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(ScoreBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.twf1 = two_weight()
        self.twf2 = two_weight()
        self.TDP1 = TDP_Attention(dim)
        self.TDP2 = TDP_Attention(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        twf1 = self.twf1(res, x)
        res1 = self.TDP1(res, x, twf1[0], twf1[1])
        res3 = self.conv2(res1)
        twf2 = self.twf2(res3, x)
        res4 = self.TDP2(res3, x, twf2[0], twf2[1])
        return res4


class Score(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, r=8, use_dropout=False, padding_type='reflect',
                 n_blocks=6):
        super(Score, self).__init__()
        # down sampling
        self.D3D = Base_Model(3, 3)
        self.D3D.load_state_dict(
            torch.load("path your best dehazing model"))
        for param in self.D3D.parameters():
            param.requires_grad = False

        self.class_model = Selector()

    def forward(self, input):
        out_u, out_f, out_fu, x_U, x_F, x_FU = self.D3D(input)
        fea_map = torch.cat((x_U.unsqueeze(1), x_F.unsqueeze(1), x_FU.unsqueeze(1)), dim=1)
        # **********************************************************************************
        score_u = self.class_model(x_U)
        score_f = self.class_model(x_F)
        score_fu = self.class_model(x_FU)
        return fea_map, score_u, score_f, score_fu, out_u, out_f, out_fu

    def Gumbel_Softmax_delrandom(self, logits, tau=1.0, hard=False, dim=-1):
        y_soft = (logits / tau).softmax(dim)
        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret


class Selector(nn.Module):
    def __init__(self, ngf=64):
        super(Selector, self).__init__()
        self.net = nn.Sequential(
            ScoreBlock(default_conv,ngf,3),
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1),
            nn.ReLU(),

            ScoreBlock(default_conv, ngf, 3),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(),

            ScoreBlock(default_conv, ngf, 3),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, padding=0),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(),

            ScoreBlock(default_conv, ngf * 2, 3),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(),

            ScoreBlock(default_conv, ngf * 2, 3),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, padding=0),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(),

            ScoreBlock(default_conv, ngf * 4, 3),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(),

            ScoreBlock(default_conv, ngf * 4, 3),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, padding=0),
            nn.InstanceNorm2d(ngf * 8),
            nn.ReLU(),

            ScoreBlock(default_conv, ngf * 8, 3),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(ngf * 8),
            nn.ReLU(),

            ScoreBlock(default_conv, ngf * 8, 3),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ngf * 8, ngf * 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ngf * 16, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        res = self.net(x).view(batch_size).unsqueeze(1)
        res = torch.tanh(res)
        return res

