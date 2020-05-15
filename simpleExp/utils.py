import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from modules import *

def getArchParams(net, MixedBlock = MixedBlock):
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

def transfer_from_ori(model_type, ori_filename):
    model = model_type()
    state_dict = model.state_dict()
    s_keys = list(state_dict.keys())
    ori_state_dict = torch.load(ori_filename)
    o_keys = list(ori_state_dict.keys())
    i = 0
    j = 0
    # print(len(s_keys))
    # print(len(o_keys))
    while (True):
        if i == len(s_keys) or j == len(o_keys):
            break
        if s_keys[i].find("mix") != -1:
            i += 1
        else:
            state_dict[s_keys[i]] = ori_state_dict[o_keys[j]]
            i += 1
            j += 1
    return state_dict