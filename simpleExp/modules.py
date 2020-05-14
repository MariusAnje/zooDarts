import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Block(nn.Module):
    # supported type: CONV1, CONV3, CONV5, CONV7, ID
    def __init__(self, bType:str, in_channels:int, out_channels:int, norm:bool = False):
        super(Block, self).__init__()
        if bType == "ID":
            self.op = nn.Identity()
        elif bType == "CONV1":
            self.op = nn.Conv2d(in_channels, out_channels, 1)
        elif bType == "CONV3":
            self.op = nn.Conv2d(in_channels, out_channels, 3, padding = 1)
        elif bType == "CONV5":
            self.op = nn.Conv2d(in_channels, out_channels, 5, padding = 2)
        elif bType == "CONV7":
            self.op = nn.Conv2d(in_channels, out_channels, 7, padding = 3)
        self.act = nn.ReLU()
        if norm:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
        
        

    def forward(self, x):

        return self.act(self.norm(self.op(x)))

class MixedBlock(nn.Module):
    def __init__(self, modules:list, in_channels, out_channels):
        super(MixedBlock, self).__init__()
        moduleList = []
        for m in modules:
            moduleList.append(Block(m, in_channels, out_channels))
        self.moduleList = nn.ModuleList(moduleList)
        self.mix = nn.Parameter(torch.Tensor(len(modules))).requires_grad_()
        self.sm = nn.Softmax(dim=0)
        

    def forward(self, x):
        p = self.sm(self.mix)
#         print(p)
        output = p[0] * self.moduleList[0](x)
        for i in range(1, len(self.moduleList)):
            output += p[i] * self.moduleList[i](x)
        return output
    
class SuperNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(SuperNet, self).__init__()
        modules = ["CONV1", "CONV3", "CONV5", "CONV7"]
        self.block1 = MixedBlock(modules, 3, 128)
        self.block2 = MixedBlock(modules, 128, 128)
        self.block3 = MixedBlock(modules, 128, 256)
        self.block4 = MixedBlock(modules, 256, 256)
        self.block5 = MixedBlock(modules, 256, 512)
        self.block6 = MixedBlock(modules, 512, 512)
        self.pool = nn.MaxPool2d(2)
        self.lastPool = nn.AdaptiveAvgPool2d((4,4))
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
        x = self.lastPool(x)
        x = self.classifier(x.view(-1, 512*4*4))
        return x
