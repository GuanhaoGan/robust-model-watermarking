import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from collections import OrderedDict

from models import CleanBatchNorm2d, ResNet
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = CleanBatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = CleanBatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)),
                ('bn', CleanBatchNorm2d(self.expansion*planes))
                ]))

    def _shortcut(self, input, mask):
        if len(self.shortcut)==0:
            return self.shortcut(input)
        else:
            return self.shortcut[1](self.shortcut[0](input), mask)

    def forward(self, x, mask=None):
        out = F.relu(self.bn1(self.conv1(x), mask))
        out = self.bn2(self.conv2(out), mask)
        out += self._shortcut(x, mask)
        out = F.relu(out)
        return out

def through_layer(layer, x, mask=None):
    for i in range(len(layer)):
        if isinstance(layer[i], CleanBatchNorm2d) or isinstance(layer[i], BasicBlock):
            x = layer[i](x, mask)
        else:
            x = layer[i](x)
    return x


class ResNetCBN(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetCBN, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = CleanBatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x, mask=None, lin=0, lout=6):
        out = x        
        if mask is None:
            mask = torch.zeros(len(x), dtype=torch.long).to(x.device)
            
        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out, mask)
            out = F.relu(out)
        if lin < 2 and lout > 0:
            out = through_layer(self.layer1,out,mask)
            # print(out.shape[0])
        if lin < 3 and lout > 1:
            out = through_layer(self.layer2,out,mask)
        if lin < 4 and lout > 2:
            out = through_layer(self.layer3,out,mask)
        if lin < 5 and lout > 3:
            out = through_layer(self.layer4,out,mask)
        if lin < 6 and lout > 4:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
        if lout > 5:
            out = self.linear(out)
        return out

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = CleanBatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = CleanBatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))
            ]))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = CleanBatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = CleanBatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = CleanBatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(OrderedDict([
                ('conv',nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)),
                ('bn',CleanBatchNorm2d(self.expansion*planes))
            ]))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

def ResNetCBN18(num_classes=10):
    return ResNetCBN(BasicBlock, [2,2,2,2], num_classes)

# test()
if __name__ == "__main__":
    model = ResNetCBN18()
    img = torch.randn(5,3,32,32)
    mask = torch.ones((5,))
    mask[-1]=0
    outputs = model(img, mask)
    print(outputs.shape)