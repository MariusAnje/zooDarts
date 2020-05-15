import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from modules import Block

class HWMixedBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, norm:bool):
        super(HWMixedBlock, self).__init__()
        self.convBlock = Block("CONV3", in_channels, out_channels, norm)
        self.mix = nn.Parameter(torch.randn(out_channels)).requires_grad_()
        self.sm = nn.Softmax(dim=-1)
        

    def forward(self, x):
        x = self.convBlock(x)
        p = 1 - self.sm(self.mix)
        output = (x.transpose(1,3) * p).transpose(1,3) * (float(len(p))/float((len(p) - 1)))
        return output
    
    def inference(self, x)
        x = self.convBlock(x)
        p = self.mix / self.mix
        p[self.mix.argmax()] *= 0
        output = (x.transpose(1,3) * p).transpose(1,3) * (float(len(p))/float((len(p) - 1)))
        return output
        
class HWnet_old(nn.Module):
    def __init__(self, num_classes = 10):
        super(HWnet, self).__init__()
        
        self.feature = nn.Sequential(
            HWMixedBlock(3,128,True),
            HWMixedBlock(128,128,True),
            nn.MaxPool2d(2),
            HWMixedBlock(128,256,True),
            HWMixedBlock(256,256,True),
            nn.MaxPool2d(2),
            HWMixedBlock(256,512,True),
            HWMixedBlock(512,512,True),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x.view(-1, 512*4*4))
        return x

class HWnet(nn.Module):
    def __init__(self, num_classes = 10):
        super(HWnet, self).__init__()
        
        self.block1 = HWMixedBlock(3,128,True)
        self.block2 = HWMixedBlock(128,128,True)
        self.block3 = HWMixedBlock(128,256,True)
        self.block4 = HWMixedBlock(256,256,True)
        self.block5 = HWMixedBlock(256,512,True)
        self.block6 = HWMixedBlock(512,512,True)
        self.pool   = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.pool(x)
        x = self.classifier(x.view(-1, 512*4*4))
        return x