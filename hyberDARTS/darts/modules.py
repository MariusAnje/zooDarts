import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import sys
import numpy as np
import os

if os.path.expanduser("~").find("afs") == -1:
    GETTQDM = False
else:
    GETTQDM = True

if GETTQDM:
    from tqdm import tqdm
else:
    class tqdm():
        def __init__(self, x):
            self.x = x
            self.happyInterestingFlag = True

        def __enter__(self):
            return self
        
        def __exit__(self ,type, value, traceback):
            pass

        def __iter__(self):
            return iter(self.x)

        def __next__(self):
            return next(self.x)

def n_para(module:nn.Module, input_size:torch.Size):
    ic = module.in_channels
    oc = module.out_channels
    ks = module.kernel_size
    return ic * oc * ks[0] * ks[1] * input_size[-1] * input_size[-2]

def quantize(x, num_int_bits, num_frac_bits, signed=True):
    precision = 1 / 2 ** num_frac_bits
    x = torch.round(x / precision) * precision
    if signed is True:
        bound = 2 ** (num_int_bits - 1)
        return torch.clamp(x, -bound, bound-precision)
    else:
        bound = 2 ** num_int_bits
        return torch.clamp(x, 0, bound-precision)

class QuantValue(nn.Module):
    """
    Quantization
    """
    def __init__(self, N, m):
        super(QuantValue, self).__init__()
        self.N = N
        self.m = m
        self.quant = QuantValue_F.apply

    def forward(self, x):
        return self.quant(x, self.N, self.m)
    
    def extra_repr(self):
        s = ('N = %d, m = %d'%(self.N, self.m))
        return s

class QuantValue_F(torch.autograd.Function):
    """
    res = clamp(round(input/pow(2,-m)) * pow(2, -m), -pow(2, N-1), pow(2, N-1) - 1)
    """

    @staticmethod 
    def forward(ctx, inputs, N, m):
        Q = pow(2, N - 1) - 1
        delt = pow(2, - m)
        M = (inputs.to(torch.float32)/delt).round().clamp(-Q-1,Q)
        return delt*M
    
    @staticmethod
    def backward(ctx, g):
        return g , None, None


class LayerBlock(nn.Module):
    """
    A block for convolution layers with activation (ReLU) and normalization
    supported type: CONV1, CONV3, CONV5, CONV7, ID
    Also calculates the latency of this block (well, neglects the latency of BN and activation layers and only count conv layers)
    """
    def __init__(self, bType:str, in_channels:int, out_channels:int, norm:bool):
        super(LayerBlock, self).__init__()
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
        self.input_size = None
        self.latency_offer = n_para # DEPRECATED
        self.latency = None         # DEPRECATED
        
    def init_latency(self):
        """
            DEPRECATED
            Initialize the latency for this block
        """
        self.latency = self.latency_offer(self.op, self.input_size)
    
    def get_latency(self, HW, name):
        """
            get the latency of this block from HW registers
        """
        # if self.latency is None:
        #     self.init_latency()
        return HW.get_latency(name, self.op, self.input_size)
    
    def forward(self, x):
        if self.input_size is None:
            self.input_size = x.size()
        return self.norm(self.act(self.op(x)))

class QuantBlock(nn.Module):
    """
    QuantBlock
    """
    def __init__(self, layer:LayerBlock, quant_params):
        super(QuantBlock, self).__init__()
        self.quant_weight = QuantValue(quant_params['weight_num_int_bits'] + quant_params['weight_num_frac_bits'], quant_params['weight_num_frac_bits'])
        self.quant_act    = QuantValue(quant_params['act_num_int_bits']    + quant_params['act_num_frac_bits'],    quant_params['act_num_frac_bits'])
        self.op = copy.deepcopy(layer.op)
        self.act = copy.deepcopy(layer.act)
        self.norm = copy.deepcopy(layer.norm)
        self.quant_params = quant_params
        self.input_size = None
    
    # def extra_repr(self):
    #     return f"(quant): Weight ({self.quant_params['weight_num_int_bits']}, {self.quant_params['weight_num_frac_bits']}), Act ({self.quant_params['act_num_int_bits']}, {self.quant_params['act_num_frac_bits']})"
    
    def get_latency(self, HW, name):
        """
            get the latency of this block from HW registers
        """

        return HW.get_latency(name, self.op, self.input_size)
    
    def forward(self, x):
        if self.input_size is None:
            self.input_size = x.size()
        if isinstance(self.op, nn.Conv2d):
            op = self.op
            weight, bias, stride, padding, dilation, groups = \
                op.weight, op.bias, op.stride, op.padding, op.dilation, op.groups
            # weight = quantize(
            #             weight,
            #             self.quant_params['weight_num_int_bits'],
            #             self.quant_params['weight_num_frac_bits'],
            #             signed=True)
            # bias = quantize(
            #             bias,
            #             self.quant_params['weight_num_int_bits'],
            #             self.quant_params['weight_num_frac_bits'],
            #             signed=True)
            weight = self.quant_weight(weight)
            bias   = self.quant_weight(bias)

            x = F.conv2d(x, weight, bias, stride, padding, dilation, groups)
            
            # x = quantize(
            #     x,
            #     self.quant_params['act_num_int_bits'],
            #     self.quant_params['act_num_frac_bits'],
            #     signed=False
            #     )
            
            x = self.norm(x)
            x = self.act(x)
            x = self.quant_act(x)
            return x
        else:
            return self.norm(self.act(self.op(x)))



class MixedHW(nn.Module):
    """
        A hardware mixed block where hosting different hardware designs
    """
    def __init__(self, num):
        super(MixedHW, self).__init__()
        self.sm = nn.Softmax(dim=-1)
        self.is_super = True
        self.mix = nn.Parameter(torch.ones(num)).requires_grad_()
        self.latency = {}
        """
            TODO: replace this random HW by functions
        """
        self.HW = np.random.randn(num)
        self.HW = self.HW - self.HW.min()
        self.HW = (self.HW/self.HW.max()) + 1
    
    def init_latency(self, name, module, input_size):
        """
            initialize the latency of one software module using different hardware designs
            TODO: replace random HW by functions
        """
        latencyItem = np.zeros(len(self.mix))
        for i in range(len(self.mix)):
            latencyItem[i] = self.HW[i] * n_para(module, input_size) # TODO: replace random HW by functions
        self.latency[name] = latencyItem
    
    def get_latency(self, name, module, input_size):
        """
            Returns the latency of one module
            it is the weighted sum of different hardware designs
        """
        if not (name in self.latency.keys()):
            self.init_latency(name, module, input_size)

        if self.is_super:
            p = self.sm(self.mix)
            output = p[0] * self.latency[name][0]
            for i in range(1, len(self.mix)):
                output += p[i] * self.latency[name][i]
            return output
        else:
            i = self.mix.argmax()
            return self.latency[name][i]

class MixedBlock(nn.Module):
    def __init__(self, modules:list):
        super(MixedBlock, self).__init__()
        self.creat_modules(modules)
        self.sm = nn.Softmax(dim=-1)
        self.is_super = True

    def creat_modules(self, modules:list):
        self.moduleList = nn.ModuleList(modules)
        self.mix = nn.Parameter(torch.ones(len(modules))).requires_grad_()

    def forward(self, x):
        """
        Weighted sum on each paths
        """
        if self.is_super:
            p = self.sm(self.mix)
            output = p[0] * self.moduleList[0](x)
            for i in range(1, len(self.moduleList)):
                output += p[i] * self.moduleList[i](x)
            return output
        # if self.is_super:
        #     p = self.sm(self.mix)
        #     m = []
        #     for i in range(0, len(self.moduleList)):
        #         m.append(p[i] * self.moduleList[i](x))
        #     output = m[0]
        #     for i in range(1, len(m)):
        #         output += m[i]
        #     return output
        else:
            i = self.mix.argmax()
            return self.moduleList[i](x)
    
    def get_latency(self, HW, name):
        if self.is_super:
            p = self.sm(self.mix)
            output = p[0] * self.moduleList[0].get_latency(HW, name = name + ".moduleList." + str(0))
            for i in range(1, len(self.moduleList)):
                output += p[i] * self.moduleList[i].get_latency(HW, name = name + ".moduleList." + str(i))
            return output
        else:
            i = self.mix.argmax()
            return self.moduleList[i].get_latency(HW, name = name + ".moduleList." + str(i))

class SuperNet(nn.Module):
    def __init__(self):
        super(SuperNet, self).__init__()
        self.model = None
        self.is_super = True
        self.HW = MixedHW(5)
    
    def get_model(self, model):
        """
            Initiate superNet model
        """
        self.model = model
    
    def load_state_dict(self, state_dict):
        """
            Load pretrained parameters
        """
        self.model.load_state_dict(state_dict)
    
    def get_arch_params(self, get_name = False):
        """
            get probabilities (logits) for architecture
            returns a list of tensors
        """
        arch_params = []
        arch_names  = []
        for name, para in self.model.named_parameters():
            if name.find(".mix") != -1:
                arch_params.append(para)
                arch_names.append(name)
        arch_params.append(self.HW.mix)
        if get_name:
            return arch_params, arch_names
        else:
            return arch_params
    
    def get_module_choice(self):
        """
            get the finally chosen module for each mixed block
            returns a list of integers 
        """
        module_choice = []
        # for name, module in self.model.named_modules():
        #     if isinstance(module, MixedBlock):
        #         module_choice.append((name, module.mix.argmax().item()))
        arch_params, arch_names = self.get_arch_params(get_name=True)
        for item in arch_params:
            module_choice.append(item.argmax().item())
        return module_choice, arch_names
    
    def get_net_params(self):
        """
            get NN parameters of the model
            returns a list of tensors
        """
        net_params = []
        for name, para in self.model.named_parameters():
            if name.find(".mix") == -1:
                net_params.append(para)
        return net_params

    def modify_super(self, is_super):
        """
            modify if self.model inference as a super net or use chosen modules
            True: acts as a super net
            False: use the module with highest probability
        """
        self.is_super = is_super
        for module in self.model.modules():
            if isinstance(module, MixedBlock) or isinstance(module, MixedHW):
                module.is_super = is_super
    
    def get_arch_loss(self, criterion, arch_outputs, arch_labels):
        """
            get the loss used to update architecture parameters
            returns a scaler
        """
        alpha = 0.1
        beta = 0
        # print(self.get_latency())
        # print(self.get_latency().log())
        # exit()
        # return - (1 - criterion(arch_outputs, arch_labels))* self.get_latency() * 1e-8
        return criterion(arch_outputs, arch_labels) * (alpha * self.get_latency().log()).pow(beta)
    
    def get_arch_loss_debug(self, criterion, arch_outputs, arch_labels):
        """
            DEBUG use
            get the loss used to update architecture parameters
            returns a scaler
        """
        return criterion(arch_outputs, arch_labels), self.get_latency() * 0#1e-10
    
    def get_latency(self):
        """
            get weighted latency for each component
            returns a scaler
        """
        latency = 0.0
        for name, module in self.model.named_children():
            # if isinstance(module, MixedBlock):
            if str(type(module)).find("MixedBlock") != -1:
                latency += module.get_latency(self.HW, name)
            if isinstance(module, nn.Sequential):
                for sub_name, sub_m in module.named_children():
                    if str(type(sub_m)).find("MixedBlock") != -1:
                        latency += sub_m.get_latency(self.HW, name+"."+sub_name)

        # print(f"{latency} !!!")
        return latency
    
    def get_unrolled_model_grad(self, plus:bool, net_grads_f, arch_inputs, arch_labels, criterion, eps):
        """
            a tool to get second-order derivatives (gradients) for the model
            grad_new = grad(criterion(model(weight + eps * singer * (grad_old), arch_inputs), arch_labels))
            returns a set of gradients
        """
        unrolled_model = copy.deepcopy(self.model)
        unrolled_model.eval()
        if plus:
            signer = 1
        else:
            signer = -1
        i = 0
        for name, param in unrolled_model.named_parameters():
            if name.find("mix") == -1:
                try:
                    param.data += (eps * net_grads_f[i] * signer)
                except:
                    pass
                i += 1
        arch_outputs = unrolled_model(arch_inputs)
        arch_loss = self.get_arch_loss(criterion, arch_outputs, arch_labels)
        arch_loss.backward()
        arch_grads_s = []
        for name, param in unrolled_model.named_parameters():
            if name.find("mix") != -1:
                arch_grads_s.append(param.grad.data)
        return arch_grads_s

    
    def unroll(self, arch_loader, arch_optimizer, net_optimizer, criterion, device):
        """
            get second-order derivatives (gradients) for the model
            grad2 = grad(loss(model(w^+, inputs), labels)) - grad(loss(model(w^-, inputs), labels))
            grad_step = grad1 - grad2/(2 * eps)
            returns a set of gradients
        """
        arch_data = next(iter(arch_loader))
        eps = 1e-5
        net_optimizer.zero_grad()
        arch_optimizer.zero_grad()
        arch_inputs, arch_labels = arch_data
        arch_inputs, arch_labels = arch_inputs.to(device), arch_labels.to(device)
        self.model.eval()
        arch_outputs = self.model(arch_inputs)
        arch_loss = self.get_arch_loss(criterion, arch_outputs, arch_labels)
        arch_loss.backward()
        # with hardware
        # arch_grads_f = [copy.deepcopy(gf.grad.data) for gf in self.get_arch_params()]

        # without hardware
        # TODO: want to see if I can change it back to gf.grad.data
        arch_grads_f = [copy.deepcopy(gf.grad.data) for gf in self.get_arch_params()[:-1]]
        arch_grads_f.append(copy.deepcopy(self.get_arch_params()[-1].grad))
        net_grads_f  = [copy.deepcopy(gf.grad.data) for gf in self.get_net_params()]
        # net_optimizer.zero_grad()
        arch_optimizer.zero_grad()
        arch_grads_s_p = self.get_unrolled_model_grad(True, net_grads_f, arch_inputs, arch_labels, criterion, eps)
        # net_optimizer.zero_grad()
        arch_optimizer.zero_grad()
        arch_grads_s_n = self.get_unrolled_model_grad(False, net_grads_f, arch_inputs, arch_labels, criterion, eps)
        arch_optimizer.zero_grad()
        faster_break = len(self.get_arch_params())
        for i, param in enumerate(self.get_arch_params()):
            if i == faster_break - 1:
                try:
                    param.grad.data = arch_grads_f[i]
                except:
                    param.grad = arch_grads_f[i]
            else:
                param.grad.data = arch_grads_f[i] - (arch_grads_s_p[i] - arch_grads_s_n[i])/(2*eps)
        return arch_loss


    def train(self, net_loader, arch_loader, arch_optimizer, net_optimizer, criterion, device):
        """
            trains the super net
        """
        self.model.train()

        loss_list = []
        avg_size = 100
        running_loss = 0.0
        running_arch_loss = 0.0
        i = 0
        # arch_loader = iter(arch_loader)
        with tqdm(net_loader) as run_loader:
            for inputs, labels in run_loader:
                self.model.train()
                inputs, labels = inputs.to(device), labels.to(device)
                net_optimizer.zero_grad()
                arch_optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss_list.append(loss.data.clone())
                running_loss += loss
                loss.backward()
                net_optimizer.step()
                i += 1
                
                # arch_data = next(iter(arch_loader))
                # net_optimizer.zero_grad()
                # arch_optimizer.zero_grad()
                # arch_inputs, arch_labels = arch_data
                # arch_inputs, arch_labels = arch_inputs.to(device), arch_labels.to(device)
                # arch_outputs = self.model(arch_inputs)
                # arch_loss = criterion(arch_outputs, arch_labels)
                # arch_loss.backward()

                arch_loss = self.unroll(arch_loader, arch_optimizer, net_optimizer, criterion, device)
                running_arch_loss += arch_loss.item()
                arch_optimizer.step()
                try:
                    run_loader.happyInterestingFlag
                except:
                    run_loader.set_description(f"{running_loss/i:.4f}, {running_arch_loss/i:.4f}")
    
    def train_short(self, net_loader, arch_loader, arch_optimizer, net_optimizer, criterion, device, stop):
        """
            trains the super net
            Used to get better understanding of memory consumption, a debug function
        """
        self.model.train()

        loss_list = []
        avg_size = 100
        running_loss = 0.0
        i = 0
        # arch_loader = iter(arch_loader)
        with tqdm(net_loader) as run_loader:
            for inputs, labels in run_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                net_optimizer.zero_grad()
                arch_optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss_list.append(loss.data.clone())
                running_loss += loss
                loss.backward()
                net_optimizer.step()
                i += 1
                try:
                    run_loader.happyInterestingFlag
                except:
                    run_loader.set_description(f"{running_loss/i:.4f}")

                self.unroll(arch_loader, arch_optimizer, net_optimizer, criterion, device)
                arch_optimizer.step()
                if i == stop:
                    break

    def train_debug(self, net_loader, arch_loader, arch_optimizer, net_optimizer, criterion, device):
        """
            trains the super net
        """
        self.model.train()

        loss_list = []
        avg_size = 100
        running_loss = 0.0
        i = 0
        # arch_loader = iter(arch_loader)
        c_gradList = []
        l_gradList = []
        with tqdm(net_loader) as run_loader:
            for inputs, labels in run_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                net_optimizer.zero_grad()
                arch_optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss_list.append(loss.data.clone())
                running_loss += loss
                loss.backward()
                net_optimizer.step()
                i += 1
                try:
                    run_loader.happyInterestingFlag
                except:
                    run_loader.set_description(f"{running_loss/i:.4f}")

                c_grad, l_grad = self.unroll_debug(arch_loader, arch_optimizer, net_optimizer, criterion, device)
                c_gradList.append(c_grad)
                l_gradList.append(l_grad)
                arch_optimizer.step()
        return c_gradList, l_gradList
    
    def unroll_debug(self, arch_loader, arch_optimizer, net_optimizer, criterion, device):
        """
            get second-order derivatives (gradients) for the model
            grad2 = grad(loss(model(w^+, inputs), labels)) - grad(loss(model(w^-, inputs), labels))
            grad_step = grad1 - grad2/(2 * eps)
            returns a set of gradients
        """
        arch_data = next(iter(arch_loader))
        eps = 1e-5
        net_optimizer.zero_grad()
        arch_optimizer.zero_grad()
        arch_inputs, arch_labels = arch_data
        arch_inputs, arch_labels = arch_inputs.to(device), arch_labels.to(device)
        arch_outputs = self.model(arch_inputs)
        c_loss, l_loss = self.get_arch_loss_debug(criterion, arch_outputs, arch_labels)
        c_loss.backward()
        c_grad_f = [copy.deepcopy(gf.grad.data) for gf in self.get_arch_params()[:-1]]
        l_loss.backward()
        arch_grads_f = [copy.deepcopy(gf.grad.data) for gf in self.get_arch_params()]
        l_grad_f = [copy.deepcopy(arch_grads_f[i].data - c_grad_f[i].data) for i in range(len(self.get_arch_params())-1)]

        
        net_grads_f  = [copy.deepcopy(gf.grad.data) for gf in self.get_net_params()]
        # net_optimizer.zero_grad()
        arch_optimizer.zero_grad()
        arch_grads_s_p = self.get_unrolled_model_grad(True, net_grads_f, arch_inputs, arch_labels, criterion, eps)
        # net_optimizer.zero_grad()
        arch_optimizer.zero_grad()
        arch_grads_s_n = self.get_unrolled_model_grad(False, net_grads_f, arch_inputs, arch_labels, criterion, eps)
        arch_optimizer.zero_grad()
        faster_break = len(self.get_arch_params())
        for i, param in enumerate(self.get_arch_params()):
            if i == faster_break - 1:
                param.grad.data = arch_grads_f[i]
            else:
                param.grad.data = arch_grads_f[i] - (arch_grads_s_p[i] - arch_grads_s_n[i])/(2*eps)
        return c_grad_f, l_grad_f
    
    def warm(self, net_loader, net_optimizer, criterion, device):
        """
            warming up, training the super net without updating arch parameters
            Note that initial probabilities for each module are the same
        """
        self.model.train()

        loss_list = []
        avg_size = 100
        running_loss = 0.0
        i = 0
        # arch_loader = iter(arch_loader)
        with tqdm(net_loader) as run_loader:
            for inputs, labels in run_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                net_optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss_list.append(loss.data.clone())
                running_loss += loss
                loss.backward()
                net_optimizer.step()
                i += 1
                try:
                    run_loader.happyInterestingFlag
                except:
                    run_loader.set_description(f"{running_loss/i:.4f}")
    
    def test(self, loader, device):
        """
            inference and return the accuracy
        """
        correct = 0
        total = 0
        ct = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                predicted = torch.argmax(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            return correct / total


        
if __name__ == "__main__":
    from models import QuantCIFARNet
    net = SuperNet()
    subspace = [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3], [1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
    net.get_model(QuantCIFARNet(subspace))
    print(net.model)
    theInput = torch.Tensor(1,3,32,32)
    o = net.model(theInput)
    o.sum().backward()
    print(net.get_arch_params())
    print(net.get_arch_params()[0].grad)
    net.modify_super(False)
    
    
    # layer = LayerBlock("CONV3", 3, 5, True)
    # quant_params = {'weight_num_int_bits':3,'weight_num_frac_bits':8, 'act_num_int_bits':3, 'act_num_frac_bits':8}
    # model1 = QuantBlock(layer, quant_params)
    # model2 = QuantBlock(layer, quant_params)
    # with torch.no_grad():
    #     model1.op.weight /= model1.op.weight
    #     # print(model1.op.weight)
    #     # print(model2.op.weight)
    # a = torch.randn(1,3,3,3)
    # print(model1(a))
    # # new