import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class two_weight(nn.Module):
    def __init__(self):
        super(two_weight, self).__init__()
        self.F = nn.Flatten(3, 4)
        # self.avg_layer = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.avg_layer = nn.AdaptiveAvgPool2d(1)
        # 1x1 Conv layer
        # self.conv = nn.Conv2d(in_channels=s_size, out_channels=1, kernel_size=1)
        self.conv = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1)
        self.soft = nn.Softmax(dim=1)
        # ReLU function
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x1 = torch.unsqueeze(x1, 1)
        x2 = torch.unsqueeze(x2, 1)
        x = self.F(torch.cat((x1, x2), 1))  # batch_size*L*C*S

        x = self.avg_layer(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.soft(x)
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


class ResnetGlobalAttention(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ResnetGlobalAttention, self).__init__()

        self.feature_channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv_end = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.soft = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)

        # 添加全局信息
        zx = y.squeeze(-1)
        zy = zx.permute(0, 2, 1)
        zg = torch.matmul(zy, zx)
        zg = zg / self.feature_channel

        batch = zg.shape[0]
        v = zg.squeeze(-1).permute(1, 0).expand((self.feature_channel, batch))
        v = v.unsqueeze_(-1).permute(1, 2, 0)

        # 全局局部信息融合
        atten = self.conv(y.squeeze(-1).transpose(-1, -2))
        atten = atten + v
        atten = self.conv_end(atten)
        atten = atten.permute(0, 2, 1).unsqueeze(-1)

        atten_score = self.soft(atten)

        return x * atten_score

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0

        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]


        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class DehazeBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(DehazeBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.twf1 = two_weight()
        self.twf2 = two_weight()
        self.TDP1 = TDP_Attention(dim)
        self.TDP2 = TDP_Attention(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        twf1 = self.twf1(res,x)
        res1 = self.TDP1(res,x,twf1[0],twf1[1])
        res3 = self.conv2(res1)
        twf2 = self.twf2(res3,x)
        res4 = self.TDP2(res3,x,twf2[0],twf2[1])
        return res4


class Base_Model(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, r=8,use_dropout=False, padding_type='reflect',
                 n_blocks=6):
        super(Base_Model, self).__init__()
        # down sampling
        self.down_pt1 = nn.Sequential(nn.ReflectionPad2d(3),
                                      nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                                      nn.InstanceNorm2d(ngf),
                                      nn.ReLU(True))
        self.down_pt2 = nn.Sequential(nn.ReflectionPad2d(3),
                                      nn.Conv2d(ngf, input_nc, kernel_size=7, padding=0),
                                      nn.InstanceNorm2d(input_nc),
                                      nn.ReLU(True))
        self.down_resize = nn.Sequential(nn.ReflectionPad2d(3),
                                         nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                                         nn.InstanceNorm2d(ngf),
                                         nn.ReLU(True))
        self.down_pt11 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
                                        nn.InstanceNorm2d(ngf * 2),
                                        nn.ReLU(True))

        self.down_pt21 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
                                       nn.InstanceNorm2d(ngf * 4),
                                       nn.ReLU(True))

        # 3DPfa_block
        self.block1 = DehazeBlock(default_conv, ngf, 3)
        self.block2 = DehazeBlock(default_conv, ngf, 3)
        self.block3 = DehazeBlock(default_conv, ngf, 3)
        self.block4 = DehazeBlock(default_conv, ngf, 3)
        self.block5 = DehazeBlock(default_conv, ngf, 3)
        self.block6 = DehazeBlock(default_conv, ngf, 3)

        norm_layer = nn.InstanceNorm2d
        activation = nn.ReLU(True)
        model_res1 = []
        for i in range(n_blocks):
            model_res1 += [
                ResnetBlock(ngf * 4, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.model_res1 = nn.Sequential(*model_res1)


        # up sampling
        self.up1 = nn.Sequential(
                                nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                nn.InstanceNorm2d(ngf * 2),
                                nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.InstanceNorm2d(ngf),
                                 nn.ReLU(True))


        self.TDP1 = TDP_Attention(ngf )
        self.TDP2 = TDP_Attention(ngf * 2)
        self.TDP3 = TDP_Attention(ngf * 4)
        self.TDP4 = TDP_Attention(ngf)
        self.tw1 = two_weight()
        self.tw2 = two_weight()
        self.tw3 = two_weight()
        self.tw4 = two_weight()

        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

        self.sc1 = nn.Sequential(nn.Conv2d(ngf,ngf//r,1,bias=True),
                                 nn.ReLU(),
                                 nn.Conv2d(ngf//r,ngf//r,1,bias=True),  # wx+b
                                 nn.Sigmoid()
                                 )
        self.sc2 = nn.Sequential(nn.Conv2d(ngf,ngf//r,1,bias=True),
                                 nn.ReLU(),
                                 nn.Conv2d(ngf//r,ngf//r,1,bias=True),  # wx+b
                                 nn.Sigmoid()
                                 )
        self.sc3 = nn.Sequential(nn.Conv2d(ngf,ngf//r,1,bias=True),
                                 nn.ReLU(),
                                 nn.Conv2d(ngf//r,ngf//r,1,bias=True),  # wx+b
                                 nn.Sigmoid()
                                 )
        self.conv = nn.Sequential( nn.Conv2d(ngf,ngf,kernel_size=3,stride=1,padding=1),
                                   nn.InstanceNorm2d(ngf),
                                   nn.ReLU(),
                                   nn.ReflectionPad2d(3),
                                   nn.Conv2d(ngf, output_nc,kernel_size=7,padding=0),
                                   nn.Tanh())

    def forward(self, input):

        #***************************************U-Net*************************************************
        x_down0 = self.down_resize(input)  # ngf
        x_down1 = self.down_pt11(x_down0)  # ngf*2
        x_down2 = self.down_pt21(x_down1)  # ngf*4
      
        x2 = self.model_res1(x_down2)

        tw1 = self.tw1(x_down2, x2)  # ngf*4
        x2 = self.TDP3(x_down2, x2, tw1[0], tw1[1])

        x21 = self.up1(x2)

        tw2 = self.tw2(x_down1, x21)
        x1 = self.TDP2(x_down1, x21, tw2[0], tw2[1])
        x10 = self.up2(x1)

        tw3 =self.tw3(x_down0,x10)
        x_U = self.TDP1(x_down0,x10,tw3[0],tw3[1])
        #***************************************U-Net*************************************************


        #**********************************************************************************
        x_down11 = self.down_pt1(input)

        # x_pt = self.model_res2(x_down11)
        x_pt = self.block1(x_down11)
        x_pt = self.block2(x_pt)
        x_pt = self.block3(x_pt)
        x_pt = self.block4(x_pt)
        x_pt = self.block5(x_pt)
        x_L = self.block6(x_pt)
        #********************************************************************************************

        #**********************************************************************************
        tw4 = self.tw4(x_U, x_L)
        x_out = self.TDP4(x_U, x_L, tw4[0], tw4[1])
        #********************************************************************************************


        #*******************************************************************************
        x_U1 = x_U
        x_L1 = x_L
        x_out1 = x_out
        #********************************************************************************************
        score_u = self.sc1(self.pool1(x_U1)).sum().unsqueeze(0)
        score_single = self.sc2(self.pool2(x_L1)).sum().unsqueeze(0)
        score_su = self.sc3(self.pool3(x_out1)).sum().unsqueeze(0)
        score_map = torch.cat((score_u,score_single,score_su),dim=0)
        fea_map = torch.cat((x_U.unsqueeze(0),x_L.unsqueeze(0),x_out.unsqueeze(0)),dim=0)
        score_idx = nn.functional.gumbel_softmax(score_map,hard=True,dim=0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(fea_map)
        #********************************************************************************************
        out_map = score_idx* fea_map
        out_map = out_map.sum(0)
        out = self.conv(out_map)
        return out

class Discriminator(nn.Module):
    def __init__(self, bn=False, ngf=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ngf, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, padding=0),
            nn.InstanceNorm2d(ngf * 2),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(ngf * 2),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, padding=0),
            nn.InstanceNorm2d(ngf * 4) if not bn else nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(ngf * 4) if not bn else nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, padding=0),
            nn.BatchNorm2d(ngf * 8) if bn else nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(ngf * 8) if bn else nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ngf * 8, ngf * 16, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ngf * 16, 1, kernel_size=1)

        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))
