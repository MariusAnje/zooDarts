import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from modules import LayerBlock, MixedBlock, QuantBlock

class SuperCIFARNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(SuperCIFARNet, self).__init__()
        modules = ["CONV7", "CONV7", "CONV7"]
        # modules = ["CONV1","CONV3", "CONV5", "CONV7"]
        norm = True
        self.block1 = MixedBlock(self.createConvList(modules, 3, 128, norm))
        self.block2 = MixedBlock(self.createConvList(modules, 128, 128, norm))
        self.block3 = MixedBlock(self.createConvList(modules, 128, 256, norm))
        self.block4 = MixedBlock(self.createConvList(modules, 256, 256, norm))
        self.block5 = MixedBlock(self.createConvList(modules, 256, 512, norm))
        self.block6 = MixedBlock(self.createConvList(modules, 512, 512, norm))
        # self.block1 = MixedBlock(self.createConvList(modules, 3, 64, norm))
        # self.block2 = MixedBlock(self.createConvList(["CONV3", "CONV7"], 64, 64, norm))
        # self.block3 = MixedBlock(self.createConvList(["CONV3", "CONV5", "CONV7"], 64, 64, norm))
        # self.block4 = MixedBlock(self.createConvList(["CONV1", "CONV3"], 64, 64, norm))
        # self.block5 = MixedBlock(self.createConvList(["CONV1", "CONV5", "CONV7"], 64, 64, norm))
        # self.block6 = MixedBlock(self.createConvList(["CONV1", "CONV3", "CONV5"], 64, 64, norm))
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

class SubCIFARNet(nn.Module):
    def __init__(self, subspace, num_classes = 10):
        super(SubCIFARNet, self).__init__()
        modules = ["CONV1", "CONV3", "CONV5", "CONV7"]
        # modules = ["CONV1","CONV3", "CONV5"]
        out_channels = [128, 128, 256, 256, 512, 512]
        # out_channels = [64, 64, 64, 64, 64, 64]
        in_channels = 3

        norm = True
        moduleList = []
        for i in range(6):
            submodules = self.parseSubSpace(modules, subspace[i])
            moduleList.append(MixedBlock(self.createConvList(submodules, in_channels, out_channels[i], norm)))
            in_channels = out_channels[i]
            if i in [1,3,5]:
                moduleList.append(nn.MaxPool2d(2))
        self.feature = nn.Sequential(*moduleList)
        self.classifier = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            # nn.Linear(64*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def createConvList(self, modules:list, in_channels:int, out_channels:int, norm:bool):
        convList = []
        for bType in modules:
            convList.append(LayerBlock(bType, in_channels, out_channels, norm))
        return convList
    
    def parseSubSpace(self, modules, space):
        subspace = []
        for item in space:
            subspace.append(modules[item])
        return subspace

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x.view(-1, 512*4*4))
        # x = self.classifier(x.view(-1, 64*4*4))
        return x

class QuantCIFARNet(nn.Module):
    def __init__(self, subspace, num_classes = 10):
        super(QuantCIFARNet, self).__init__()
        modules = ["CONV1", "CONV3", "CONV5", "CONV7"]
        quant_params = [{'weight_num_int_bits':3,'weight_num_frac_bits':8, 'act_num_int_bits':3, 'act_num_frac_bits':8},
        {'weight_num_int_bits':1,'weight_num_frac_bits':10, 'act_num_int_bits':1, 'act_num_frac_bits':10}]
        # modules = ["CONV1","CONV3", "CONV5"]
        out_channels = [128, 128, 256, 256, 512, 512]
        # out_channels = [64, 64, 64, 64, 64, 64]
        in_channels = 3

        norm = True
        moduleList = []
        for i in range(6):
            submodules = self.parseSubSpace(modules, subspace[i])
            moduleList.append(MixedBlock(self.createConvList(submodules, quant_params, in_channels, out_channels[i], norm)))
            in_channels = out_channels[i]
            if i in [1,3,5]:
                moduleList.append(nn.MaxPool2d(2))
        self.feature = nn.Sequential(*moduleList)
        self.classifier = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            # nn.Linear(64*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def createConvList(self, modules:list, quant:list, in_channels:int, out_channels:int, norm:bool):
        convList = []
        for i in range(len(modules)):
            quantList = []
            for j in range(len(quant)):
                quantList.append(QuantBlock(LayerBlock(modules[i], in_channels, out_channels, norm), quant[j]))
            convList.append(MixedBlock(quantList))
        return convList
    
    def parseSubSpace(self, modules, space):
        subspace = []
        for item in space:
            subspace.append(modules[item])
        return subspace

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x.view(-1, 512*4*4))
        # x = self.classifier(x.view(-1, 64*4*4))
        return x

class ChildCIFARNet(nn.Module):
    def __init__(self, rollout, num_classes = 10):
        super(ChildCIFARNet, self).__init__()
        modules = ["CONV1", "CONV3", "CONV5", "CONV7"]
        # modules = ["CONV1","CONV3", "CONV5"]
        out_channels = [128, 128, 256, 256, 512, 512]
        # out_channels = [64, 64, 64, 64, 64, 64]

        in_channels = 3

        norm = True
        moduleList = []
        for i in range(6):
            moduleList.append(LayerBlock(modules[rollout[i]], in_channels, out_channels[i], norm))
            in_channels = out_channels[i]
            if i in [1,3,5]:
                moduleList.append(nn.MaxPool2d(2))
        self.feature = nn.Sequential(*moduleList)
        self.classifier = nn.Sequential(
            # nn.Linear(512*4*4, 1024),
            nn.Linear(64*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        # x = self.classifier(x.view(-1, 512*4*4))
        x = self.classifier(x.view(-1, 64*4*4))
        return x

class QuantChildCIFARNet(nn.Module):
    def __init__(self, rollout, quant_params, num_classes = 10):
        super(QuantChildCIFARNet, self).__init__()
        modules = ["CONV1", "CONV3", "CONV5", "CONV7"]
        # modules = ["CONV1","CONV3", "CONV5"]
        out_channels = [128, 128, 256, 256, 512, 512]
        # out_channels = [64, 64, 64, 64, 64, 64]

        in_channels = 3

        norm = True
        moduleList = []
        for i in range(6):
            moduleList.append(QuantBlock(LayerBlock(modules[rollout[i]], in_channels, out_channels[i], norm), quant_params[i]))
            in_channels = out_channels[i]
            if i in [1,3,5]:
                moduleList.append(nn.MaxPool2d(2))
        self.feature = nn.Sequential(*moduleList)
        self.classifier = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            # nn.Linear(64*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x.view(-1, 512*4*4))
        # x = self.classifier(x.view(-1, 64*4*4))
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

if __name__ == "__main__":
    subspace = [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3], [1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
    model = QuantCIFARNet(subspace)
    print(model)
    x = torch.randn(16,3,32,32)
    y = model(x).sum()
    y.backward()