import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def getArchParams(net):
    arch_Params = []
    for m in net.modules():
        if isinstance(m, MixedBlock):
            arch_Params.append(m.mix)
    return arch_Params

def getNetParams(net):
    net_Params = []
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            net_Params.append(m.weight)
            net_Params.append(m.bias)
    return net_Params