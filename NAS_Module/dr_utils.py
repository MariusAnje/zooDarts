from torch import nn
import torch
import sys
sys.path.append("../Interface")
from utility import is_same
import pattern_kernel
import copy_conv2d
import torch.nn.functional as F
from dr_modules import MixedBlock


def get_last_attr_idx(model,seq):

    last_not_digit = 0
    pre_attr = model
    last_attr = []


    a = model
    for idx in range(len(seq)):
        var = seq[idx]
        if var.isdigit():
            a = a[int(var)]
        else:
            pre_attr = a
            a = getattr(a, var)
            last_not_digit = idx
            last_attr = a
    return pre_attr,last_attr,last_not_digit


def make_mixed(model,layer, layer_name,num_modules):

    # print("Debug:")
    # print("name:",layer_name)
    
    # [M, N, K, S, G, P, b] = (
    #     layer.out_channels, layer.in_channels, is_same(layer.kernel_size),
    #     is_same(layer.stride), layer.groups, is_same(layer.padding), layer.bias)
        
    seq = layer_name.split(".")
    (pre_attr,last_attr,last_not_digit) = get_last_attr_idx(model, seq)

    ## Weiwen: 03-29
    ## Step 2: Backup weights and bias if exist
    ##
    """
    is_b = False
    if type(b)==nn.Parameter:
        ori_para_w = model.state_dict()[layer_name + ".weight"][:]
        ori_para_b = model.state_dict()[layer_name + ".bias"][:]
        is_b = True
    else:
        ori_para_w = model.state_dict()[layer_name + ".weight"][:]
    """
    
    new_conv = MixedBlock(layer, num_modules)
    if last_not_digit == len(seq) - 1:
        # last one is the attribute, directly setattr
        setattr(pre_attr, seq[-1], new_conv)
    elif last_not_digit == len(seq) - 2:
        # one index last_attr[]
        last_attr[int(seq[-1])] = new_conv
        setattr(pre_attr, seq[-2], last_attr)
    elif last_not_digit == len(seq) - 3:
        # two index last_attr[][]
        last_attr[int(seq[-2])][int(seq[-1])] = new_conv
        setattr(pre_attr, seq[-3], last_attr)
    else:
        print("more than 2 depth of index from last layer is not support!")
        sys.exit(0)

    ## Weiwen: 03-29
    ## Step 4: Setup new parameters from backup
    ##
    """
    if is_b:
        model.state_dict()[layer_name + ".bias"][:] = ori_para_b

    if var_k>0:
        pad_fun = torch.nn.ZeroPad2d(int(var_k/2))
        model.state_dict()[layer_name + ".weight"][:] = pad_fun(ori_para_w)
    """


    return model


