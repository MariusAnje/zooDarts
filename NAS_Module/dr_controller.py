# %%

import logging
import csv
import numpy as np
# import tensorflow as tf
import sys
from rl_input import controller_params, HW_constraints
import termplotlib as tpl
import copy
import random
from datetime import datetime
import time
import torch
import os

import train
import utils

from pattern_generator import pattern_sets_generate_3
from model_search_space import ss_mnasnet1_0, ss_mnasnet0_5, ss_resnet18, ss_mobilenet_v2, ss_proxyless_mobile

logger = logging.getLogger(__name__)


def ema(values):
    """
    Helper function for keeping track of an exponential moving average of a list of values.
    For this module, we use it to maintain an exponential moving average of rewards
    """
    weights = np.exp(np.linspace(-1., 0., len(values)))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode="full")[:len(values)]
    return a[-1]


class Controller(object):

    def __init__(self):
        self.args = train.parse_args()

        self.data_loader,self.data_loader_test = train.get_data_loader(self.args)

        self.alpha = float(self.args.alpha)
        self.target_acc = [float(x) for x in self.args.target_acc.split(" ")]
        self.target_lat = [float(x) for x in self.args.target_lat.split(" ")]
        

        [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p] = [int(x.strip()) for x in self.args.cconv.split(",")]
        self.HW = [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p]
        self.HW2 = [int(x.strip()) for x in self.args.dconv.split(",")]
        """
        self.graph = tf.Graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config, graph=self.graph)

        self.hidden_units = controller_params['hidden_units']
        """
        
        if self.args.model == "resnet18":
            self.nn_model_helper = ss_resnet18
        elif self.args.model == "mnasnet0_5":
            self.nn_model_helper = ss_mnasnet0_5
        elif self.args.model == "mobilenet_v2":
            self.nn_model_helper = ss_mobilenet_v2
        elif self.args.model == "proxyless_mobile":
            self.nn_model_helper = ss_proxyless_mobile
        elif self.args.model == "mnasnet1_0":
            self.nn_model_helper = ss_mnasnet1_0

        
        space_name = self.nn_model_helper.get_space()[0]
        space = self.nn_model_helper.get_space()[1]

        self.nn1_search_space = space
        # self.hw1_search_space = controller_params['hw_space']

        self.nn1_num_para = len(self.nn1_search_space)
        # self.hw1_num_para = len(self.hw1_search_space)


        self.num_para = self.nn1_num_para #+ self.hw1_num_para

        self.nn1_beg, self.nn1_end = 0, self.nn1_num_para
        # self.hw1_beg, self.hw1_end = self.nn1_end, self.nn1_end + self.hw1_num_para

        self.para_2_val = {}
        idx = 0
        for hp in self.nn1_search_space:
            self.para_2_val[idx] = hp
            idx += 1

        # for hp in self.hw1_search_space:
        #     self.para_2_val[idx] = hp
        #     idx += 1

        """
        self.RNN_classifier = {}
        self.RNN_pred_prob = {}
        with self.graph.as_default():
            self.build_controller()

        self.reward_history = []
        self.architecture_history = []
        self.trained_network = {}

        self.explored_info = {}

        self.target_HW_Eff = HW_constraints["target_HW_Eff"]
        """
        
        self.pattern_space = pattern_sets_generate_3((3,3))
        

    def global_train(self):
        import torchvision
        from dr_modules import make_mixed
        import dr_modules
        from torch import nn
        import torch.optim as optim
        from tqdm import tqdm
        from model_search_space import ss_resnet18

        """
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
        
        step = 0
        total_rewards = 0
        child_network = np.array([[0] * self.num_para], dtype=np.int64)
        """
        device = torch.device(self.args.device)
        
        space = self.nn1_search_space
        model = torchvision.models.__dict__[self.args.model](pretrained=True)
        
        # use preset DNAs to create pruned model
        dna = [23, 14, 8, 42, 0, 128, 256, 256, 480, 496, 16, 16, 16, 16, 8, 8, 8, 8, 2, -1, 0]
        pat_point, exp_point, ch_point, quant_point, comm_point = dna[0:4], dna[4], dna[5:10], dna[10:18], dna[18:21]
        HW = copy.deepcopy(self.HW)
        HW[5] += comm_point[0]
        HW[6] += comm_point[1]
        HW[7] += comm_point[2]
        model = self.nn_model_helper.resnet_18_dr_pre_dna(model, pat_point, exp_point, ch_point, quant_point, self.args)

        # search non-preset hyperparameters: search space
        pattern_idx, k_expand, ch_list, q_list, comm_point = space[0:4], space[4], space[5:10], space[10:18], space[18:21]
        layer_names, layer_kernel_inc, channel_cut_layers, quant_layers, quant_paras \
            = self.nn_model_helper.resnet_18_dr(pattern_idx, k_expand, ch_list, q_list, self.args)

        # create model
        training = True # if train (finetune) the pruned the model
        mixedModel = dr_modules.MixedResNet18(model, HW, device)

        #calculate original latency
        mixedModel.to(device)
        mixedModel.get_ori_latency(device, quant_layers[4:])

        # create DARTS model
        mixedModel.to(torch.device("cpu"))
        # mixedModel.device = torch.device("cpu")
        mixedModel.create_mixed_prune(layer_names, layer_kernel_inc, channel_cut_layers[4:], quant_layers[4:], quant_paras, self.args)
        mixedModel.to(device)
        arch_params = mixedModel.get_arch_params()
        net_params = mixedModel.get_net_params()
        arch_optimizer = optim.Adam(arch_params, lr = 1e-5)
        net_optimizer = optim.Adam(net_params, lr = 1e-5)
        criterion = nn.CrossEntropyLoss()


        if self.args.pretrained:
            training = False
            state_dict = torch.load(self.args.checkpoint)
            mixedModel.load_state_dict(state_dict)
            """
            print(mixedModel.get_arch_params())
            for name, module in mixedModel.model.named_modules():
                if name == quant_layers[4]:
                    print(module)
                    for m in module.moduleList:
                        print(m.quan_paras)
            """

        # print(model)
        if training:
            # mixedModel.train_fast(self.data_loader, arch_optimizer, net_optimizer, criterion, device, 2000)
            mixedModel.modify_super(False)
            mixedModel.train_fast(self.data_loader, self.data_loader_test, arch_optimizer, net_optimizer, criterion, device, -1, self.args, logging)
            # mixedModel.train_fast(self.data_loader, net_optimizer, net_optimizer, criterion, device, 200, self.args)


        mixedModel.modify_super(True)
        acc = mixedModel.test(self.data_loader_test, device)
        here_latency = mixedModel.get_latency(device)#.detach().cpu().numpy()
        ori_latency = mixedModel.ori_latency
        logging.info(f"acc: {acc:.4f}, train latency: {here_latency:.4f}, total: {ori_latency + here_latency:.4f}")
        arch_params_ori = mixedModel.get_arch_params()
        arch_params_print = []
        for param in arch_params_ori:
            arch_params_print.append(param.data.cpu().numpy())
        logging.info(f"arch parameters: {arch_params_print}")
        exit()

        """
        ====================================================================================================

        ends here

        ====================================================================================================
        """

# %%


seed = 0
torch.manual_seed(seed)
random.seed(seed)
fileHandler = logging.FileHandler("log_dr_exp", mode = "a+")
fileHandler.setLevel(logging.INFO)
streamHandler = logging.StreamHandler()
streamHandler.setLevel(logging.DEBUG)
logging.basicConfig(
                    handlers=[
                                fileHandler,
                                streamHandler],
                    level= logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s')

logging.info("============== Begin ==============")
# logging.debug("debug")
controller = Controller()
controller.global_train()

# %%
