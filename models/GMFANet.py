import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .__init__ import *
from torch.nn.parameter import Parameter


class MultiScaleAttention(nn.Module):
    def __init__(self, ops=2):
        super(MultiScaleAttention, self).__init__()
        self.alpha1 = Parameter(torch.Tensor(1,1))
        self.alpha2 = Parameter(torch.Tensor(1,1))
        self.ops = ops
        torch.nn.init.xavier_normal_(self.alpha1.data)
        torch.nn.init.xavier_normal_(self.alpha2.data)
    def forward(self, x):
        if self.ops==2:
            x1, x2 =x # x1 is unsampled, x2 is extracted
            if F.sigmoid(self.alpha1) < 0.12:
                mask = 0.0
            else:
                mask = F.sigmoid(self.alpha1)
            x1_out = mask*x1
            x2_out = (1-mask)*x2
            x_out = x1_out+x2_out
        else:
            sample_index = torch.randint(x.shape[0]-2, (1,))
            x_out = (1-F.sigmoid(self.alpha1))*x[sample_index[0]]+F.sigmoid(self.alpha1)*x[-2]
            x_out = (1-F.sigmoid(self.alpha2))*x_out+F.sigmoid(self.alpha2)*x[-1]
        return x_out, (F.sigmoid(self.alpha1), F.sigmoid(self.alpha2))



class GMFANet(nn.Module):
    def __init__(self, n_classes, device, pretrained=True):
        super(GMFANet, self).__init__()
        self.feature_dim = 512
        self.base = resnet34(pretrained=pretrained, n_classes=n_classes)


        self.avg_pool2 = nn.AdaptiveAvgPool2d((28, 28))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.MultiScaleAttention2 = MultiScaleAttention(ops=2)

        self.avg_pool3 = nn.AdaptiveAvgPool2d((14, 14))
        self.conv3 = nn.Conv2d(128, 256, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv1_3 = nn.Conv2d(64, 256, kernel_size=1)
        self.bn1_3 = nn.BatchNorm2d(256)
        self.MultiScaleAttention3 = MultiScaleAttention(ops=-1)

        self.avg_pool4 = nn.AdaptiveAvgPool2d((7, 7))
        self.conv4 = nn.Conv2d(256, 512, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv1_4 = nn.Conv2d(64, 512, kernel_size=1)
        self.bn1_4 = nn.BatchNorm2d(512)
        self.conv2_4 = nn.Conv2d(128, 512, kernel_size=1)
        self.bn2_4 = nn.BatchNorm2d(512)
        self.MultiScaleAttention4 = MultiScaleAttention(ops=-1)

        self.target_att = torch.ones(7, 7).to(device)

        self.FAM = FAM(512, num_classes = n_classes).to(device)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, n_classes)
        self.tau = 0.0


    def forward(self, x):
        x1_in = self.base.get_features(x)

        x2_in = self.base.layer2(x1_in)

        x2 = self.avg_pool2(x1_in)
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x2, alpha2 = self.MultiScaleAttention2(torch.cat((torch.unsqueeze(x2,dim=0), torch.unsqueeze(x2_in,dim=0)),dim=0))
        x3_in = self.base.layer3(x2)

        x1_3 = self.avg_pool3(x1_in)
        x1_3 = F.relu(self.bn1_3(self.conv1_3(x1_3)))
        x3 = self.avg_pool3(x2)
        x3 = F.relu(self.bn3(self.conv3(x3)))
        x3, alpha3 = self.MultiScaleAttention3(torch.cat((torch.unsqueeze(x1_3,dim=0), torch.unsqueeze(x3,dim=0), torch.unsqueeze(x3_in,dim=0)),dim=0))
        x4_in = self.base.layer4(x3)

        x1_4 = self.avg_pool4(x1_in)
        x1_4 = F.relu(self.bn1_4(self.conv1_4(x1_4)))
        x2_4 = self.avg_pool4(x2)
        x2_4 = F.relu(self.bn2_4(self.conv2_4(x2_4)))
        x4 = self.avg_pool4(x3)
        x4 = F.relu(self.bn4(self.conv4(x4)))
        x4, alpha4 = self.MultiScaleAttention4(torch.cat((torch.unsqueeze(x1_4, dim=0),
                                                                      torch.unsqueeze(x2_4, dim=0),
                                                                      torch.unsqueeze(x4, dim=0),
                                                                      torch.unsqueeze(x4_in, dim=0)), dim=0))
        att4, _ = self.FAM(x4)
        if self.tau != 1:
            att_norm = torch.maximum(att4, self.target_att)
        else:
            att_norm = self.target_att
        att_norm[att_norm>self.tau] = 1.0
        x4 = x4 * att_norm
        out = x4
        out = self.avgpool(out)
        out_ = torch.flatten(out, 1)
        out = self.fc(out_)
        return out, att_norm

class FAM(nn.Module):
    def __init__(self, in_channels, num_classes = 21):
        super(FAM, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, num_classes, 7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(num_classes, in_channels, 7, stride=1, padding=3)
        self.bn0 = nn.BatchNorm2d(num_classes)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.unsample = nn.UpsamplingBilinear2d((224, 224))


    def forward(self, x):
        x_out_ = F.relu(self.bn0(self.conv0(x)))
        x_out_grad = F.relu(self.bn1(self.conv1(x_out_)))
        x_out = x_out_grad.mean(dim=(2, 3), keepdim=True) * x
        out = x_out.sum(dim=1, keepdim=True)
        out[out< 0] = 0
        out = (out - out.min())/(1e-7 + out.max())
        out_up = self.unsample(out)
        return out, out_up


