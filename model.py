import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, drop_rate, batch_norm=True, num_bottleneck=256, linear=True):
        super(ClassBlock, self).__init__()
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if batch_norm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if drop_rate > 0:
            add_block += [nn.Dropout(p=drop_rate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.add_block = add_block

        classifier = [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        return self.classifier(x)


class PCB(nn.Module):
    def __init__(self, class_num):
        super(PCB, self).__init__()

        self.part = 6
        self.model = models.resnet50(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)

        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

        # define classifier for each part
        for i in range(self.part):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(2048, class_num, drop_rate=0.5))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)

        # get six part feature batchsize*2048*6
        part, predict = {}, {}
        for i in range(self.part):
            part[i] = x[:, :, i].view(x.size(0), x.size(1))
            name = 'classifier' + str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])

        return list(predict.values())


'''
debug model structure, run this code with:
$ python model.py
'''
if __name__ == '__main__':
    net = PCB(500)
    print(net)
    input = Variable(torch.FloatTensor(32, 3, 256, 128))
    print('input shape:', input.shape)
    outputs = net(input)
    print('output shape:', len(outputs), 'parts x', outputs[0].shape)
