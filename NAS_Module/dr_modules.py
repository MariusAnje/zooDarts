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
            norm = False
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
        self.drop = nn.Dropout(p = 0.3)
        
        

    def forward(self, x):

        return self.norm(self.act(self.op(x)))
    
    def inference(self, x):
        
        return self.forward(x)

class MixedBlock(nn.Module):
    def __init__(self, module, module_num):
        super(MixedBlock, self).__init__()
        if isinstance(module, nn.Conv2d):
            I, O, K, S, P, D, G, B, M = module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, module.bias != None, module.padding_mode
            moduleList = []
            for m in range(module_num):
                new_conv = nn.Conv2d(I, O, K, S, P, D, G, B, M)
                new_conv.weight.data = module.weight.data
                if B:
                    new_conv.bias.data = module.bias.data
                moduleList.append(new_conv)
        elif isinstance(module, nn.BatchNorm2d):
            ch = module.num_features
            [eps, momentum, affine, track_running_stats] = (module.eps, module.momentum, module.affine, module.track_running_stats)
            moduleList = []
            for _ in range(module_num):
                new_bn = nn.BatchNorm2d(ch, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
                new_bn.weight.data = module.weight.data
                new_bn.bias.data = module.bias.data
                new_bn.running_mean.data = module.running_mean.data
                new_bn.running_var.data = module.running_var.data 
                new_bn.num_batches_tracked.data = module.num_batches_tracked.data
                # new_bn.eval()
                moduleList.append(new_bn)
        else:
            moduleList = []
            new_thing = module
            new_thing.weight.data = module.weight.data
            new_thing.bias.data = module.bias.data
            moduleList.append(new_thing)
        self.moduleList = nn.ModuleList(moduleList)
        self.mix = nn.Parameter(torch.ones(module_num)).requires_grad_()
        self.sm = nn.Softmax(dim=-1)
        

    def forward(self, x):
        # return self.moduleList[0](x)
        p = self.sm(self.mix)
#         print(p)
        output = p[0] * self.moduleList[0](x)
        for i in range(1, len(self.moduleList)):
            output += p[i] * self.moduleList[i](x)
        return output
    
    def superEval(self, x):
        i = self.mix.argmax()
        return self.moduleList[i](x)
    
class SuperNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(SuperNet, self).__init__()
        # modules = ["CONV1", "CONV3", "CONV5", "CONV7"]
        modules = ["CONV1","CONV3", "CONV5"]
        norm = True
        self.block1 = MixedBlock(modules, 3, 128, norm)
        self.block2 = MixedBlock(modules, 128, 128, norm)
        self.block3 = MixedBlock(modules, 128, 256, norm)
        self.block4 = MixedBlock(modules, 256, 256, norm)
        self.block5 = MixedBlock(modules, 256, 512, norm)
        self.block6 = MixedBlock(modules, 512, 512, norm)
        self.pool = nn.MaxPool2d(2)
        self.lastPool = nn.MaxPool2d(2)
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
        
