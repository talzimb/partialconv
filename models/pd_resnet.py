###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#   
# Copyright (c) 2017, Soumith Chintala. All rights reserved.
###############################################################################
'''
Code adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
Introduced partial convolutoins based padding for convolutional layers
'''

import torch.nn as nn
import math
import torch
import torch.utils.model_zoo as model_zoo
from .partialconv2d import PartialConv2d

__all__ = ['PDResNet', 'pdresnet18', 'pdresnet34', 'pdresnet50', 'pdresnet101',
           'pdresnet152']


# model_urls = {
#     'pdresnet18': '',
#     'pdresnet34': '',
#     'pdresnet50': '',
#     'pdresnet101': '',
#     'pdresnet152': '',
# }

model_urls = {
    'pdresnet18': '',
    'pdresnet34': '',
    'pdresnet50': '',
    'pdresnet101': '',
    'pdresnet152': '',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return PartialConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = PartialConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = PartialConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = PartialConv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PDResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(PDResNet, self).__init__()
        self.conv1 = PartialConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # self.fc = nn.Linear(512 * block.expansion, 2) #num classes = 2

        self.fc1 = nn.Linear(512 * block.expansion, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 2)

        for m in self.modules():
            if isinstance(m, PartialConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                PartialConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, mask):

        x = self.conv1(x, mask_in=mask)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        last_conv_layer = torch.flatten(x, 1)
        last_layer = self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(last_conv_layer))))))
        x = self.fc4(last_layer)

        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x, last_layer


def pdresnet18(pretrained=False, **kwargs):
    """Constructs a PDResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PDResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['pdresnet18']))
    return model


def pdresnet34(pretrained=False, **kwargs):
    """Constructs a PDResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PDResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['pdresnet34']))
    return model


def pdresnet50(pretrained=False, **kwargs):
    """Constructs a PDResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    def init_classifier(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)

    model = PDResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # TODO find a place for pretrained models
        # PATH = '/home/eligol/Documents/01_WIS/partial_conv/partialconv/experiment_1/checkpoint_pdresnet50_multigpu_b16/pdresnet50.pth'
        # PATH = os.path.join(os.getcwd(), 'pretrained_checkpoints/pdresnet50.pth')
        PATH = '/home/projects/yonina/SAMPL_training/covid_partialconv/pretrained_checkpoints/pdresnet50.pth'
        # Initialize model
        model.apply(init_classifier)
        model_dict = model.state_dict()
        checkpoint = torch.load(PATH)
        # model.load_state_dict(model_zoo.load_url(model_urls['pdresnet50']))
        checkpoint_dict = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}

        # remove class output layers because their different shape and update weights from checkpoint
        del checkpoint_dict['fc.weight']
        del checkpoint_dict['fc.bias']
        model_dict.update(checkpoint_dict)
        model.load_state_dict(model_dict)
    return model


def pdresnet101(pretrained=False, **kwargs):
    """Constructs a PDResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PDResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['pdresnet101']))
    return model


def pdresnet152(pretrained=False, **kwargs):
    """Constructs a PDResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PDResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['pdresnet152']))
    return model
