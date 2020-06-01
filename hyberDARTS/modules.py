import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LayerBlock(nn.Module):
    """
    A block for convolution layers with activation (ReLU) and normalization
    supported type: CONV1, CONV3, CONV5, CONV7, ID
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
        
    def forward(self, x):
        return self.norm(self.act(self.op(x)))

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
        else:
            i = self.mix.argmax()
            return self.moduleList[i](x)

class SuperNet(nn.Module):
    def __init__(self):
        super(SuperNet, self).__init__()
        self.model = None
        self.is_super = True
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
    
    def get_arch_params(self):
        arch_params = []
        for name, para in self.model.named_parameters():
            if name.find(".mix") != -1:
                arch_params.append(para)
        return arch_params
    
    def get_net_params(self):
        net_params = []
        for name, para in self.model.named_parameters():
            if name.find(".mix") == -1:
                net_params.append(para)
        return net_params

    def modify_super(self, is_super):
        for module in self.model.modules():
            if isinstance(module, MixedBlock):
                module.is_uper = is_super
                self.is_super = is_super
    
    def get_model(self, model):
        self.model = model

    def forward(self, x):
        return self.model(x)

        
if __name__ == "__main__":
    from models import SuperCIFARNet
    net = SuperNet()
    net.get_model(SuperCIFARNet())
    theInput = torch.Tensor(1,3,32,32)
    o = net(theInput)
    o.sum().backward()
    print(net.get_arch_params())
    print(net.get_arch_params()[0].grad)