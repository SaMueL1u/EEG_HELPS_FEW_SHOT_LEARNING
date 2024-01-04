import torch.nn as nn
import torch
from model.ResNet50 import ResNet50


def l2_penalty(w0, w1, alpha):
    return torch.sum((w0 - w1).pow(2)) / 2 * alpha


# 针对18/34层网络的conv2_x，conv3_x，conv4_x，conv5_x的系列卷积层
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

# 针对50/101/152层网络的conv2_x，conv3_x，conv4_x，conv5_x的系列卷积层
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

# 网络ResNet
class ResNet(nn.Module):
    def __init__(self, Path, block, blocks_num, num_classes=1000, include_top=True):
        # block有两种：BasicBlock针对18/34层网络，Bottleneck针对50/101/152层网络
        # blocks_num是一个列表，表示conv2_x，conv3_x，conv4_x，conv5_x分别对应的卷积层个数
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
													padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, blocks_num[0], stride=1)   # conv2_x
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)  # conv3_x
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)  # conv4_x
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)  # conv5_x
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        model = ResNet50(num_classes=1000)
        model = model.cuda()
        model.load_state_dict(torch.load(Path), strict=False)
        self.result_model= {}
        for param_tensor in model.state_dict():
            if param_tensor != "fc.weight" and param_tensor != "fc.bias":
                self.result_model[param_tensor] = model.state_dict()[param_tensor]# 字典的遍历默认是遍历 key，所以param_tensor实际上是键值

    # 构建conv2_x，conv3_x，conv4_x，conv5_x卷积层
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None  # 设定不是虚线,downsample不为None即是虚线
        # 网络结构中虚线路径的设定，只有18/34层网络的conv2_x不执行if语句
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1,
                                                        stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        # conv2_x，conv3_x，conv4_x，conv5_x的第一个卷积层
        # 第一层是虚线路径，传入downsample，因为两个block里面默认downsample = None
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion
        # conv2_x，conv3_x，conv4_x，conv5_x每个系列的剩余卷积层，均为实线
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def checkPenalty(self, findLayer):
        for param_tensor in findLayer.state_dict():
            if param_tensor in self.result_model.keys():
                findLayer.state_dict()[param_tensor] = l2_penalty(findLayer.state_dict()[param_tensor], self.result_model[param_tensor])
        return findLayer

    def forward(self, x):

        self.conv1 = self.checkPenalty(self.conv1)
        x = self.conv1(x)  # conv1
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # maxpool
        self.layer1 = self.checkPenalty(self.layer1)
        x = self.layer1(x)  # conv2_x
        self.layer2 = self.checkPenalty(self.layer2)
        x = self.layer2(x)  # conv3_x
        self.layer3 = self.checkPenalty(self.layer3)
        x = self.layer3(x)  # conv4_x
        self.layer4 = self.checkPenalty(self.layer4)
        x = self.layer4(x)  # conv5_x
        if self.include_top:
            x = self.avgpool(x)  # avgpool
            x = torch.flatten(x, 1)
            x = self.fc(x)  # fc

        return x




def ResNet50_l1(path, num_classes=1000, include_top=True):
    return ResNet(path, Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


