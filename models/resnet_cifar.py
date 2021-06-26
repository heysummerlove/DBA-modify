'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.simple import SimpleNet
from torch.autograd import Variable
import datetime


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class _Quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, step):
        ctx.step = step.item()
        output = torch.round(input / ctx.step)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone() / ctx.step
        return grad_input, None

class quantized_conv(nn.Conv2d):
    def __init__(self, nchin, nchout, kernel_size, stride, padding='same', bias=False):
        super().__init__(in_channels=nchin, out_channels=nchout, kernel_size=kernel_size, padding=padding,
                         stride=stride, bias=False)
        # self.N_bits = 7
        # step = self.weight.abs().max()/((2**self.N_bits-1))
        # self.step = nn.Parameter(torch.Tensor([step]), requires_grad = False)

    def forward(self, input):
        self.N_bits = 7
        step = self.weight.abs().max() / ((2 ** self.N_bits - 1))
        quantize1 = _Quantize.apply
        QW = quantize1(self.weight, step)

        return F.conv2d(input, QW * step, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

class ResNet(SimpleNet):
    def __init__(self, block, num_blocks, num_classes=10,name=None, created_time=None):
        super(ResNet, self).__init__(name, created_time)
        self.in_planes = 64

        self.conv1 = quantized_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.m = nn.MaxPool2d(5, stride=5)
        # self.lin = nn.Linear(64*6*6,1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = bilinear(512 * block.expansion, num_classes)
        # self.l=nn.Parameter(torch.cuda.FloatTensor([0.0]), requires_grad=True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out1 = out.view(out.size(0), -1)
        out = self.linear(out1)
        return out


class bilinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features)
        # self.N_bits = 7
        # step = self.weight.abs().max()/((2**self.N_bits-1))
        # self.step = nn.Parameter(torch.Tensor([step]), requires_grad = False)
        # self.weight.data = quantize(self.weight, self.step).data.clone()

    def forward(self, input):
        self.N_bits = 7
        step = self.weight.abs().max() / ((2 ** self.N_bits - 1))
        quantize1 = _Quantize.apply
        QW = quantize1(self.weight, step)

        return F.linear(input, QW * step, self.bias)

def ResNet18_local():
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    return ResNet(BasicBlock, [2,2,2,2],name='Local',
                                   created_time=current_time)

def ResNet18_target():
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    return ResNet(BasicBlock, [2,2,2,2],name='target',
                                   created_time=current_time)

def ResNet34(name=None, created_time=None):
    return ResNet(BasicBlock, [3,4,6,3],name='{0}_ResNet_34'.format(name), created_time=created_time)

def ResNet50(name=None, created_time=None):
    return ResNet(Bottleneck, [3,4,6,3],name='{0}_ResNet_50'.format(name), created_time=created_time)

def ResNet101(name=None, created_time=None):
    return ResNet(Bottleneck, [3,4,23,3],name='{0}_ResNet'.format(name), created_time=created_time)

def ResNet152(name=None, created_time=None):
    return ResNet(Bottleneck, [3,8,36,3],name='{0}_ResNet'.format(name), created_time=created_time)


if __name__ == '__main__':

    net = ResNet18_local()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())
