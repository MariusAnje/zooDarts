import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from modules import *

def getArchParams(net):
    arch_Params = []
    for m in net.modules():
        if isinstance(m, MixedBlock):
            arch_Params.append(m.mix)
    return arch_Params

def getNetParams(net):
    net_Params = []
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
            for p in m.parameters():
                net_Params.append(p)
            
    return net_Params

def archSTD(arch_Params):
    loss = 0
    for param in arch_Params:
        # print(param.data.var(), param.data)
        loss += (nn.Softmax(dim=-1)(param)).var()
    
    return -loss