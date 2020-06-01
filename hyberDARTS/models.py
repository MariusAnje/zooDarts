import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from modules import LayerBlock, MixedBlock

class SuperCIFARNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(SuperCIFARNet, self).__init__()
        # modules = ["CONV1", "CONV3", "CONV5", "CONV7"]
        modules = ["CONV1","CONV3", "CONV5"]
        norm = True
        self.block1 = MixedBlock(self.createConvList(modules, 3, 128, norm))
        self.block2 = MixedBlock(self.createConvList(modules, 128, 128, norm))
        self.block3 = MixedBlock(self.createConvList(modules, 128, 256, norm))
        self.block4 = MixedBlock(self.createConvList(modules, 256, 256, norm))
        self.block5 = MixedBlock(self.createConvList(modules, 256, 512, norm))
        self.block6 = MixedBlock(self.createConvList(modules, 512, 512, norm))
        self.pool = nn.MaxPool2d(2)
        self.lastPool = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def createConvList(self, modules:list, in_channels:int, out_channels:int, norm:bool):
        convList = []
        for bType in modules:
            convList.append(LayerBlock(bType, in_channels, out_channels, norm))
        return convList
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.lastPool(x)
        x = self.classifier(x.view(-1, 512*4*4))
        return x
    
    def superEval(self, x):
        x = self.block1.superEval(x)
        x = self.block2.superEval(x)
        x = self.pool(x)
        x = self.block3.superEval(x)
        x = self.block4.superEval(x)
        x = self.pool(x)
        x = self.block5.superEval(x)
        x = self.block6.superEval(x)
        x = self.lastPool(x)
        x = self.classifier(x.view(-1, 512*4*4))
        return x

class OriNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(OriNet, self).__init__()
        
        self.feature = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
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