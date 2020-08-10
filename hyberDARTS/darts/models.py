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

        # quant_params = [{'weight_num_int_bits':8,'weight_num_frac_bits':8, 'act_num_int_bits':8, 'act_num_frac_bits':8},
        # {'weight_num_int_bits':7,'weight_num_frac_bits':9, 'act_num_int_bits':7, 'act_num_frac_bits':9}]
        out_channels = [128, 128, 256, 256, 512, 512]
        in_channels = 3
        arch_params, quant_params = self.getParams(subspace)
        norm = True
        moduleList = []
        for i in range(6):
            submodules = self.parseSubSpace(modules, arch_params[i])
            moduleList.append(MixedBlock(self.createConvList(submodules, quant_params[i], in_channels, out_channels[i], norm)))
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
    
    def getParams(self, subspace):
        num_layers = len(subspace)//5
        int_choice =  (1,3)
        # frac_choice = (1,3,6)
        # frac_choice = (6,7,8)
        frac_choice = (3,6)
        arch_params = []
        quant_params = []
        quant_keys = ['weight_num_int_bits','weight_num_frac_bits', 'act_num_int_bits', 'act_num_frac_bits']
        for i in range(num_layers):
            start = i * 5
            arch_params.append(subspace[start])
            w_i_s = subspace[start + 1]
            w_f_s = subspace[start + 2]
            a_i_s = subspace[start + 3]
            a_f_s = subspace[start + 4]
            layer_quant_params = []
            for wi in range(len(w_i_s)):
                for wf in range(len(w_f_s)):
                    for ai in range(len(a_i_s)):
                        for af in range(len(a_f_s)):
                            new_quant = {}
                            new_quant[quant_keys[0]] = int_choice[w_i_s[wi]]
                            new_quant[quant_keys[1]] = frac_choice[w_f_s[wf]]
                            new_quant[quant_keys[2]] = int_choice[a_i_s[ai]]
                            new_quant[quant_keys[3]] = frac_choice[a_f_s[af]]
                            layer_quant_params.append(new_quant)
            quant_params.append(layer_quant_params)
        return arch_params, quant_params

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
    def __init__(self, rollout, num_classes = 10):
        super(QuantChildCIFARNet, self).__init__()
        modules = ["CONV1", "CONV3", "CONV5", "CONV7"]
        num_layers = len(rollout)//5
        int_choice =  (1,3)
        # frac_choice = (1,3,6)
        frac_choice = (3,6)
        arch_params = []
        quant_params = []
        quant_keys = ['weight_num_int_bits','weight_num_frac_bits', 'act_num_int_bits', 'act_num_frac_bits']
        for i in range(num_layers):
            start = i * 5
            arch_params.append(rollout[start])
            w_i_s = rollout[start + 1]
            w_f_s = rollout[start + 2]
            a_i_s = rollout[start + 3]
            a_f_s = rollout[start + 4]
            new_quant = {}
            new_quant[quant_keys[0]] = int_choice[w_i_s]
            new_quant[quant_keys[1]] = frac_choice[w_f_s]
            new_quant[quant_keys[2]] = int_choice[a_i_s]
            new_quant[quant_keys[3]] = frac_choice[a_f_s]
            quant_params.append(new_quant)
        out_channels = [128, 128, 256, 256, 512, 512]
        # out_channels = [64, 64, 64, 64, 64, 64]

        in_channels = 3

        norm = True
        moduleList = []
        for i in range(6):
            moduleList.append(QuantBlock(LayerBlock(modules[arch_params[i]], in_channels, out_channels[i], norm), quant_params[i]))
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
    # subspace = [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3], [1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
    # model = QuantCIFARNet(subspace)
    # print(model)
    rollout = [1,1,2,1,2,1,1,2,1,2,1,1,2,1,2,1,1,2,1,2,1,1,2,1,2,1,1,2,1,2,]
    model = QuantChildCIFARNet(rollout)
    print(model)
    x = torch.randn(16,3,32,32)
    y = model(x).sum()
    y.backward()