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
        sys.path.append("../Interface")
        import bottleneck_conv_only

        """
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
        
        step = 0
        total_rewards = 0
        child_network = np.array([[0] * self.num_para], dtype=np.int64)
        """
        device = torch.device(self.args.device)
        
        space = self.nn1_search_space
        new_stuff = [
                        [23, 14,  8, 42], 
                        [31, 24, 10, 13],
                        [50, 27,  5, 23],
                        [13, 16, 50,  6],
                        [14,  9,  3, 30],
                        [47, 15, 21, 38],
                        [22, 18, 47,  9],
                        [55, 45, 35, 39],
                        [34, 38, 16, 32],
                        [24, 15, 36, 14],
                        [46, 31, 41, 22]
                    ]
        space = [new_stuff] + list(space)[4:]
        model = torchvision.models.__dict__[self.args.model](pretrained=True)
        
        # DNA found by weiwen
        dna = [23, 14, 8, 42, 0, 128, 256, 256, 480, 496, 16, 16, 16, 16, 8, 8, 8, 8, 2, -1, 0]
        # use preset DNAs to create pruned model
        found_paras = [0, 4, 1, 1, 3, 0, 1, 0, 1, 0, 1, 0, 1, 3, 1, 0, 1, 0, 1, 0, 1, 0]
        m_1 = found_paras[4]
        m_1 = found_paras[5 + m_1 * 2: 5 + m_1 * 2 + 2]
        m_2 = found_paras[13]
        m_2 = found_paras[14 + m_2 * 2: 14 + m_2 * 2 + 2]
        dna = space[0][found_paras[0]] + [0, 128, space[3][found_paras[2]], space[4][found_paras[3]], space[5][found_paras[4]], space[6][found_paras[13]], 16, 16, 16, 16, space[11][m_1[0]], space[12][m_1[1]], space[13][m_2[0]], space[14][m_2[1]], 2, -1, 0]
        logging.info(f"Found DNA: {dna}")
        pat_point, exp_point, ch_point, quant_point, comm_point = dna[0:4], dna[4], dna[5:10], dna[10:18], dna[18:21]
        HW = copy.deepcopy(self.HW)
        HW[5] += comm_point[0]
        HW[6] += comm_point[1]
        HW[7] += comm_point[2]
        model = self.nn_model_helper.resnet_18_dr_finetune(model, pat_point, exp_point, ch_point, quant_point, self.args)

        # search non-preset hyperparameters: search space
        pattern_idx, k_expand, ch_list, q_list, comm_point = space[0:4], space[4], space[5:10], space[10:18], space[18:21]
        layer_names, layer_kernel_inc, channel_cut_layers, quant_layers, quant_paras \
            = self.nn_model_helper.resnet_18_dr(pattern_idx, k_expand, ch_list, q_list, self.args)

        # create model
        training = True # if train (finetune) the pruned the model
        mixedModel = dr_modules.MixedResNet18(model, HW, device)

        #calculate original latency
        mixedModel.to(device)
        mixedModel.get_ori_latency(device, [])
        lat = bottleneck_conv_only.get_performance(model, HW[0], HW[1], HW[2], HW[3],
                                                                 HW[4], HW[5], HW[6], HW[7], device)
        print(f"{lat:.4f}")

        net_params = mixedModel.get_net_params()
        net_optimizer = optim.Adam(net_params, lr = self.args.netLr)
        criterion = nn.CrossEntropyLoss()
        for i in range(10):
            mixedModel.finetune(self.data_loader, net_optimizer, criterion, device, self.args, logging)

            acc = mixedModel.test(self.data_loader_test, device)
            ori_latency = mixedModel.ori_latency
            logging.info(f"epoch {i}: acc: {acc:.4f}, latency: {ori_latency:.4f}")
        exit()

        """
        logging.debug(f"caching test data")
        cached_test_loader = []
        for data in tqdm(self.data_loader_test, leave = False):
            cached_test_loader.append(data)
        for i in range(1,4):
            logging.info(f"For three_{i}")
            for j in range(500, 10000, 500):
                filename = f"useful_models/three_{i}/ep_{j}_dr_checkpoint_new.pt"
                state_dict = torch.load(filename)
                mixedModel.load_state_dict(state_dict)
                arch_params_ori = mixedModel.get_arch_params()
                arch_params_print = []
                for param in arch_params_ori:
                    arch_params_print.append(param.data.argmax().item())
                logging.info(f"iter: {j:-6d}, arch parameters: {arch_params_print}")
                mixedModel.modify_super(True)
                test_acc = mixedModel.test(cached_test_loader, device)
                latency = mixedModel.get_latency(device)
                logging.info(f"              SELECTED: test_acc: {test_acc:.4f}, latency: {latency + self.ori_latency:.4f}")
                mixedModel.modify_super(False)
                test_acc = mixedModel.test(cached_test_loader, device)
                latency = mixedModel.get_latency(device)
                logging.info(f"              SUPERNET: test_acc: {test_acc:.4f}, latency: {latency + self.ori_latency:.4f}")
        exit()
        """
        

        """
        ====================================================================================================

        ends here

        ====================================================================================================
        """

# %%


seed = 0
torch.manual_seed(seed)
random.seed(seed)
fileHandler = logging.FileHandler("log_finetune", mode = "a+")
fileHandler.setLevel(logging.INFO)
streamHandler = logging.StreamHandler()
streamHandler.setLevel(logging.DEBUG)
logging.basicConfig(
                    handlers=[
                                fileHandler,
                                streamHandler],
                    level= logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s')

logging.info("=" * 45 + "\n" + " " * (20 + 33) + "Begin" +  " " * 20 + "\n" + " " * 33 + "=" * 45)
# logging.debug("debug")
controller = Controller()
logging.info(f"config: lr: arch <- {controller.args.archLr}, net <- {controller.args.netLr}")
logging.info(f"weight loss: 1, weight latency: 1, 10 randoms")
controller.global_train()

# %%
