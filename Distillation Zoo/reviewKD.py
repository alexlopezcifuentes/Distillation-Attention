import math
import pdb
import torch.nn.functional as F
from torch import nn
import torch

#from .mobilenetv2 import mobile_half
# from .shufflenetv1 import ShuffleV1
# from .shufflenetv2 import ShuffleV2
# from .resnet_cifar import build_resnet_backbone, build_resnetx4_backbone
#from .vgg import build_vgg_backbone
# from .wide_resnet_cifar import wrn

import torch
from torch import nn
import torch.nn.functional as F

import resnet
import mobilenetv2


class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                    nn.Conv2d(mid_channel*2, 2, kernel_size=1),
                    nn.Sigmoid(),
                )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None):
        n,_,h,w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (shape,shape), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * z[:,0].view(n,1,h,w) + y * z[:,1].view(n,1,h,w))
        # output
        y = self.conv2(x)
        return y, x


class ReviewKD(nn.Module):
    def __init__(
        self, student, in_channels, out_channels, mid_channel
    ):
        super(ReviewKD, self).__init__()
        self.shapes = [1,7,14,28,56]
        self.student = student

        abfs = nn.ModuleList()

        for idx, in_channel in enumerate(in_channels):
            abfs.append(ABF(in_channel, mid_channel, out_channels[idx], idx < len(in_channels)-1))


        self.abfs = abfs[::-1]

    def forward(self, x):
        logit, student_features = self.student(x)
        student_features[-1] = torch.unsqueeze(torch.unsqueeze(student_features[-1], dim=2), dim=2)

        x = student_features[::-1]
        results = []
        out_features, res_features = self.abfs[0](x[0])
        results.append(out_features)
        for features, abf, shape in zip(x[1:], self.abfs[1:], self.shapes[1:]):
            out_features, res_features = abf(features, res_features, shape)
            results.insert(0, out_features)

        return logit, results


def build_review_kd(CONFIG):
    if CONFIG['MODEL']['ARCH'].lower() in ['resnet18', 'resnet34']:
        student = resnet.model_dict[CONFIG['MODEL']['ARCH'].lower()](pretrained=CONFIG['MODEL']['PRETRAINED'],
                                                                     num_classes=CONFIG['DATASET']['N_CLASSES'],
                                                                     multiscale=CONFIG['DISTILLATION']['MULTISCALE'])
        # Number of channels
        in_channels = [64, 128, 256, 512, 512]
        # out_channels = [64, 128, 256, 512, 512]
        out_channels = [256, 512, 1024, 2048, 2048]
        mid_channel = 512

    elif CONFIG['MODEL']['ARCH'].lower() == 'mobilenetv2':
        student = mobilenetv2.mobilenet_v2(pretrained=CONFIG['MODEL']['PRETRAINED'],
                                           num_classes=CONFIG['DATASET']['N_CLASSES'],
                                           multiscale=CONFIG['DISTILLATION']['MULTISCALE'])

        # in_channels = [128, 256, 512, 1024, 1024]
        in_channels = [24, 32, 96, 1280, 1280]
        out_channels = [256, 512, 1024, 2048, 2048]
        mid_channel = 256
    else:
        print('Model not properly defined in Review method')
        assert False

    model = ReviewKD(student, in_channels, out_channels, mid_channel)

    return model


class HCL(nn.Module):
    """
    Review KD Implementation. HCL as class to match the rest of algorithms.
    """

    def __init__(self):
        super(HCL, self).__init__()

    def forward(self, fstudent, fteacher):

        fteacher[-1] = torch.unsqueeze(torch.unsqueeze(fteacher[-1], dim=2), dim=2)

        loss_all = 0.0

        for fs, ft in zip(fstudent, fteacher):
            n, c, h, w = fs.shape
            loss = F.mse_loss(fs, ft, reduction='mean')
            cnt = 1.0
            tot = 1.0
            for l in [4,2,1]:
                if l >=h:
                    continue
                tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
                tmpft = F.adaptive_avg_pool2d(ft, (l,l))
                cnt /= 2.0
                loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
                tot += cnt
            loss = loss / tot
            loss_all = loss_all + loss
        return loss_all