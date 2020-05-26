import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
import sys
sys.path.append("../Interface")
from utility import is_same
import pattern_kernel
import copy_conv2d
from pattern_generator import pattern_sets_generate_3
import model_modify
import copy_conv2d
from tqdm import tqdm
import dr_hw
import numpy as np

def avg_4_list(the_input:list, avg_size:int = 0):
    if avg_size > 0:
        the_input = the_input[-avg_size:]
    # the_sum = 0.0
    # for item in the_input:
    #     the_sum += item
    return torch.Tensor(the_input).mean()


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



class MixedBlock(nn.Module):
    def __init__(self, module, module_num):
        super(MixedBlock, self).__init__()
        self.top = True
        moduleList = []
        for _ in range(module_num):
            m = deepcopy(module)
            if isinstance(m, MixedBlock):
                m.top = False
            moduleList.append(m)
                
        self.moduleList = nn.ModuleList(moduleList)
        self.mix = nn.Parameter(torch.ones(module_num)).requires_grad_()
        self.sm = nn.Softmax(dim=-1)
        self.is_super = False
        self.input_size = 0
        self.latency = 0
        
    def modify_super(self, super:bool):
        self.is_super = super
    
    def init_latency(self, Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p, device):
        latency = torch.zeros_like(self.mix)
        leaf = not isinstance(self.moduleList[0], MixedBlock)
        if leaf:
            for i in range(latency.size(0)):
                if isinstance(self.moduleList[i], nn.Conv2d) or isinstance(self.moduleList[i], copy_conv2d.Conv2d_Custom):
                    latency[i] = dr_hw.get_performance_layer(self.moduleList[i], Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p,device, size = self.input_size)
                else:
                    latency[i] = 0
        else:
            for i in range(latency.size(0)):
                latency[i] = self.moduleList[i].get_latency(Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p, device)
        
        self.latency = latency.detach().cpu().numpy()
    
    def get_latency(self):
        
        if self.is_super:
            i = self.mix.argmax()
            return self.latency[i]
        else:
            p = self.sm(self.mix)
            # print(self.latency)
            return p.dot(torch.Tensor(self.latency).to(p.device))
        
    def forward(self, x):
        self.input_size = x.size()
        if self.is_super:
            return self.superEval(x)
        else:
            p = self.sm(self.mix)
            output = p[0] * self.moduleList[0](x)
            for i in range(1, len(self.moduleList)):
                o_item = self.moduleList[i](x)
                # print()
                output += p[i] * o_item
            return output
    
    def superEval(self, x):
        i = self.mix.argmax()
        return self.moduleList[i](x)


class MixedNet(nn.Module):
    def __init__(self, model, HW, device):
        super(MixedNet, self).__init__()
        self.model = model
        self.HW = HW
        self.device = device
    
    def forward(self, x):
        self.model(x)
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def init_latency(self, device):
        HW = self.HW
        # TODO: may be bug here!!!!!
        for name, module in self.model.named_modules():
            if isinstance(module, MixedBlock):
                module.init_latency(HW[0], HW[1], HW[2], HW[3], HW[4], HW[5], HW[6], HW[7], device)

    def get_latency(self, device):
        latency = 0
        HW = self.HW
        # TODO: may be bug here!!!!!
        for name, module in self.model.named_modules():
            if isinstance(module, MixedBlock) and module.top:
                if module.top:
                    latency += module.get_latency()
        return latency

    def get_arch_params(self):
        arch_params = []
        for name, module in self.model.named_modules():
            if isinstance(module, MixedBlock):
                arch_params.append(module.mix)
        return arch_params
    
    def get_net_params(self):
        net_params = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, copy_conv2d.Conv2d_Custom) or isinstance(module, nn.BatchNorm2d):
                net_params.append(module.weight)
                try:
                    a = module.bias.size()
                    net_params.append(module.bias)
                except:
                    pass
        return net_params

    def modify_super(self, if_super):
        for name, module in self.model.named_modules():
            if isinstance(module, MixedBlock):
                module.modify_super(if_super)
    
    def train(self, loader, arch_optimizer, net_optimizer, criterion, device, num_iters):
        self.model.train()
        running_loss = 0.0
        i = 0
        run_loader = tqdm(loader)
        for inputs, labels in run_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            net_optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            net_optimizer.step()
            running_loss += loss
            

            arch_inputs, arch_labels = next(iter(loader))
            arch_inputs, arch_labels = arch_inputs.to(device), arch_labels.to(device)
            arch_optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels) + self.get_latency(device) + self.ori_latency
            loss.backward()
            arch_optimizer.step()
            
            i += 1
            run_loader.set_description(f"{running_loss/i:.4f}")

            if i%10 == 0:
                torch.save(self.model.state_dict(), "dr_checkpoint.pt")

            if i == num_iters:
                break

    def train_fast(self, loader, test_loader, arch_optimizer, net_optimizer, criterion, device, num_iters, args, logging):
        self.model.train()
        # self.model.eval()
        
        logging.debug(f"caching test data")
        cached_test_loader = []
        for data in tqdm(test_loader, leave = False):
            cached_test_loader.append(data)

        loss_list = []
        avg_size = 100
        running_loss = 0.0
        i = 0
        with tqdm(loader) as run_loader:
            for inputs, labels in run_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                net_optimizer.zero_grad()
                arch_optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                latency = self.get_latency(device)
                loss_list.append(loss.data.clone())
                running_loss += loss
                if i%2 == 0:
                    loss.backward()
                    net_optimizer.step()
                else:
                    loss += latency * 0.01 # + self.ori_latency
                    loss.backward()
                    # print(i, self.get_arch_params()[0].grad)
                    # for param in self.get_arch_params():
                    #     param.data -= param.grad.data * 1e-3
                    arch_optimizer.step()
                

                i += 1
                run_loader.set_description(f"{avg_4_list(loss_list, avg_size):.4f}, {latency:.4f}")
                
                if i%10 == 0:
                    torch.save(self.model.state_dict(), args.checkpoint)
                
                if i % 500 == 0:
                    self.modify_super(True)
                    test_acc = self.test(cached_test_loader, device)
                    logging.debug(f"test_acc: {test_acc}")
                    torch.save(self.model.state_dict(), f"ep_{i}_" + args.checkpoint)
                    self.modify_super(False)
                    self.model.train()

                if i == num_iters * 2:
                    break

    def test(self, loader, device):
        correct = 0
        total = 0
        ct = 0
        self.model.eval()
        with torch.no_grad():
            with tqdm(loader, leave = False) as loader:
                for inputs, labels in loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self.model(inputs)
                    predicted = torch.argmax(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    loader.set_description(f"{correct}, {total}")

                return correct / total


class MixedResNet18(MixedNet):
    def __init__(self, model, HW, device):
        super(MixedResNet18, self).__init__(model, HW, device)
        # self.model = model
        self.ori_latency = None

    def get_ori_latency(self, device=None, ignored_layers = []):
        model = self.model
        HW = self.HW
        self.ori_latency = dr_hw.get_performance(model, HW[0], HW[1], HW[2], HW[3], HW[4], HW[5], HW[6], HW[7], device, ignored_layers)
    
    def create_mixed_pattern(self, layer_names, layer_kernel_inc, channel_cut_layers, quant_layers, quant_paras, args):
        model = self.model
        module_dict = dict(model.named_modules())
        pattern_space = pattern_sets_generate_3((3,3))
        
        # print(len(channel_cut_layers)*3 + len(quant_layers) + len(layer_names))
        # Kernel Pattern layers
        for name in layer_names:
            module_dict = dict(model.named_modules())
            make_mixed(model, module_dict[name], name, 56)
            pattern = {}
            simple_names = []
            for i in range(56): 
                pattern[i] = pattern_space[i].reshape((3, 3))
                simple_names.append(name + f".moduleList.{i}")

            model_modify.Kernel_Patter(model, simple_names, pattern, args)
    
    def create_mixed_quant(self, layer_names, layer_kernel_inc, channel_cut_layers, quant_layers, quant_paras_ori, args):
        model = self.model
        module_dict = dict(model.named_modules())
        
        # print(len(channel_cut_layers)*3 + len(quant_layers) + len(layer_names))
        # Kernel Pattern layers
        for name in quant_layers:
            para = quant_paras_ori[name]
            if len(para[1]) > 1:
                make_mixed(model, module_dict[name], name, len(para[1]))
                
                quant_paras = {}
                simple_names = []
                for j in range(len(para[1])):
                    simple_name = name + f".moduleList.{j}"
                    quant_paras[simple_name] = [para[0], para[1][j], para[2]]
                    simple_names.append(simple_name)
                model_modify.Kenel_Quantization(model, simple_names, quant_paras)
        
        self.model = model
        self.model.to(self.device)
        self.model(torch.Tensor(1,3,224,224).to(self.device))
        self.init_latency(self.device)
    
    def create_mixed_quant_prune(self, layer_names, layer_kernel_inc, channel_cut_layers, quant_layers, quant_paras_ori, args):
        model = self.model
        module_dict = dict(model.named_modules())
        
        # print(len(channel_cut_layers)*3 + len(quant_layers) + len(layer_names))
        # Kernel Pattern layers
        """
        for name in quant_layers:
            para = quant_paras_ori[name]
            if len(para[1]) > 1:
                make_mixed(model, module_dict[name], name, len(para[1]))
                
                quant_paras = {}
                simple_names = []
                for j in range(len(para[1])):
                    simple_name = name + f".moduleList.{j}"
                    quant_paras[simple_name] = [para[0], para[1][j], para[2]]
                    simple_names.append(simple_name)
                model_modify.Kenel_Quantization(model, simple_names, quant_paras)
        """
        module_dict = dict(model.named_modules())
        
        layers_to_cut = []
        for ch_list in channel_cut_layers:
            sp = ch_list[0].split(".")
            mixed_name = sp[0] + "." + sp[1]

            make_mixed(model, module_dict[mixed_name], mixed_name, len(ch_list[3][1]))
            module_dict = dict(model.named_modules())
            print(module_dict[mixed_name])
            exit()
            
            quant_paras = {}
            simple_names = []
            for j in range(len(para[1])):
                simple_name = name + f".moduleList.{j}"
                quant_paras[simple_name] = [para[0], para[1][j], para[2]]
                simple_names.append(simple_name)
            model_modify.Kenel_Quantization(model, simple_names, quant_paras)

        

        """
        module_dict = dict(model.named_modules())
        # Channel Cut
        for item in channel_cut_layers[3:]:
            for name in item[:3]:
                make_mixed(model, module_dict[name], name, len(item[3][1]))
        
        module_dict = dict(model.named_modules())
        for name in quant_layers:
            make_mixed(model, module_dict[name], name, len(quan_paras[name][1]))
        """

        self.model = model
        self.model.to(self.device)
        self.model(torch.Tensor(1,3,224,224).to(self.device))
        self.init_latency(self.device)


"""    
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
        
class MixedBlock_old(nn.Module):
    def __init__(self, module, module_num):
        super(MixedBlock, self).__init__()
        if isinstance(module, nn.Conv2d):
            I, O, K, S, P, D, G, B, M = module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, module.bias != None, module.padding_mode
            moduleList = []
            for _ in range(module_num):
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
                
        self.moduleList = nn.ModuleList(moduleList)
        self.mix = nn.Parameter(torch.ones(module_num)).requires_grad_()
        self.sm = nn.Softmax(dim=-1)
        

    def forward(self, x):
        # return self.moduleList[0](x)
        p = self.sm(self.mix)
        output = p[0] * self.moduleList[0](x)
        for i in range(1, len(self.moduleList)):
            output += p[i] * self.moduleList[i](x)
        return output
    
    def superEval(self, x):
        i = self.mix.argmax()
        return self.moduleList[i](x)
"""
