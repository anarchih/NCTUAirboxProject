import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch


model_urls = {
    'resnet20': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class VBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(VBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out)

        return out


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


class ResNet(nn.Module):

    def __init__(self, block, layers, in_length):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d((15, 8), stride=(1, 1))
        self.fc = nn.Linear(64 * block.expansion, 32)
        # for key in self.state_dict():
            # if key.split('.')[-1] == 'weight':
                # if 'conv' in key:
                    # torch.nn.init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                # if 'bn' in key:
                    # self.state_dict()[key][...] = 1
            # elif key.split('.')[-1] == 'bias':
                # self.state_dict()[key][...] = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class TestModel(nn.Module):

    def __init__(self):
        super(TestModel, self).__init__()
        self.resnet1 = ResNet(BasicBlock, [3, 3, 3], 1)
        # self.resnet2 = ResNet(BasicBlock, [3, 3, 3], 1)
        # self.resnet3 = ResNet(BasicBlock, [3, 3, 3], 1)
        self.fc1 = nn.Linear(31, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)


        self.fc5 = nn.Linear(64, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc6 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.fc7 = nn.Linear(64, 32)
        self.bn7 = nn.BatchNorm1d(32)
        self.fc8 = nn.Linear(32, 1)


    def forward(self, cur, other):
        cur = self.resnet1(cur)
        # prd = self.resnet2(prd)
        # trd = self.resnet3(trd)

        other = self.bn1(self.relu(self.fc1(other)))
        other = self.bn2(self.relu(self.fc2(other)))
        other = self.bn3(self.relu(self.fc3(other)))
        other = self.bn4(self.relu(self.fc4(other)))

        result = torch.cat([cur, other], 1)

        result = self.bn5(self.relu(self.fc5(result)))
        result = self.bn6(self.relu(self.fc6(result)))
        result = self.bn7(self.relu(self.fc7(result)))
        result = self.fc8(result)
        return result


class FrameModel(nn.Module):
    def __init__(self):
        super(FraneModel, self).__init__()
        self.resnet1 = ResNet(BasicBlock, [3, 3, 3], 1)
        self.fc1 = nn.Linear(31, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)




    def forward(self, cur, other):
        # Encoder
        cur = self.resnet1(cur)
        other = self.bn1(self.relu(self.fc1(other)))
        other = self.bn2(self.relu(self.fc2(other)))
        other = self.bn3(self.relu(self.fc3(other)))
        other = self.bn4(self.relu(self.fc4(other)))

        vec = torch.cat([cur, other], 1)
        vec = vec.view(vec.size(0), 64, 1, 1)

        # Decoder

        return result

def resnet20(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet56(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [18, 18, 18], **kwargs)
    return model


def v20(pretrained=False, **kwargs):
    model = ResNet(VBlock, [3, 3, 3], **kwargs)
    return model


def v56(pretrained=False, **kwargs):
    model = ResNet(VBlock, [9, 9, 9], **kwargs)
    return model


def v110(pretrained=False, **kwargs):
    model = ResNet(VBlock, [18, 18, 18], **kwargs)
    return model
