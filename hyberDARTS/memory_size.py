import argparse
import csv
import logging
import os
import time
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import os
# import tqdm

import sys
sys.path.append('./darts')
sys.path.append('./RL')

from RL import controller_nl
from RL import child
from RL import data
from RL import backend
from RL.controller import Agent
from RL.config import ARCH_SPACE, QUAN_SPACE, CLOCK_FREQUENCY
from RL.utility import BestSamples
from RL.fpga.model import FPGAModel
from RL import utility

from darts import modules
from darts.modules import SuperNet, MixedBlock
from darts.models import SubCIFARNet, ChildCIFARNet

import utils
import finetune
import subprocess



# def get_args():
parser = argparse.ArgumentParser('Parser User Input Arguments')
parser.add_argument(
    'mode',
    default='nas',
    choices=['nas', 'darts', 'all', 'joint', 'nested', 'quantization'],
    help="supported dataset including : 1. nas (default), 2. joint"
    )
parser.add_argument(
    '-d', '--dataset',
    default='CIFAR10',
    help="supported dataset including : 1. MNIST, 2. CIFAR10 (default)"
    )
parser.add_argument(
    '-l', '--layers',
    type=int,
    default=6,
    help="the number of child network layers, default is 6"
    )
parser.add_argument(
    '-rl', '--rLUT',
    type=int,
    default=1e5,
    help="the maximum number of LUTs allowed, default is 10000")
parser.add_argument(
    '-rt', '--rThroughput',
    type=float,
    default=1000,
    help="the minumum throughput to be achieved, default is 1000")
parser.add_argument(
    '-e', '--epochs',
    type=int,
    default=30,
    help="the total epochs for model fitting, default is 30"
    )
parser.add_argument(
    '-ep', '--episodes',
    type=int,
    default=2000,
    help='''the number of episodes for training the policy network, default
        is 2000'''
    )
parser.add_argument(
    '-ep1', '--episodes1',
    type=int,
    default=1000,
    help='''the number of episodes for training the architecture, default
        is 1000'''
    )
parser.add_argument(
    '-ep2', '--episodes2',
    type=int,
    default=200,
    help='''the number of episodes for training the quantization, default
        is 500'''
    )
parser.add_argument(
    '-lr', '--learning_rate',
    type=float,
    default=0.2,
    help="learning rate for updating the controller, default is 0.2")
parser.add_argument(
    '-ns', '--no_stride',
    action='store_true',
    help="include stride in the architecture space, default is false"
    )
parser.add_argument(
    '-np', '--no_pooling',
    action='store_true',
    help="include max pooling in the architecture space, default is false"
    )
parser.add_argument(
    '-b', '--batch_size',
    type=int,
    default=128,
    help="the batch size used to train the child CNN, default is 128"
    )
parser.add_argument(
    '-s', '--seed',
    type=int,
    default=1,
    help="seed for randomness, default is 0"
    )
parser.add_argument(
    '-g', '--gpu',
    type=int,
    default=0,
    help="in single gpu mode the id of the gpu used, default is 0"
    )
parser.add_argument(
    '-k', '--skip',
    action='store_true',
    help="include skip connection in the architecture, default is false"
    )
parser.add_argument(
    '-a', '--augment',
    action='store_true',
    help="augment training data"
    )
parser.add_argument(
    '-m', '--multi-gpu',
    action='store_true',
    help="use all gpus available, default false"
    )
parser.add_argument(
    '-v', '--verbosity',
    type=int,
    choices=range(3),
    default=0,
    help="verbosity level: 0 (default), 1 and 2 with 2 being the most verbose"
    )

parser.add_argument('--batchSize', action="store", type=int, default=64)
parser.add_argument('--pretrained', action="store_true")
parser.add_argument('--train_epochs', action="store", type = int, default = 10)
parser.add_argument('--log_filename', action="store", type = str, default = "./experiment/log")
parser.add_argument('--device', action="store", type = str, default = "cuda:0")
parser.add_argument('--ex_info', action="store", type = str, default = "nothing special")
parser.add_argument('--rollout_filename', action="store", type = str, default = "./experiment/rollout")
parser.add_argument('--method', action="store", type = str, default = "comp")
parser.add_argument('--wsSize', action="store", type = int, default = 9)
args = parser.parse_args()
args.device = f"cuda:{args.gpu}"
args.batchSize = args.batch_size


if args.no_stride is True:
    if 'stride_height' in ARCH_SPACE:
        ARCH_SPACE.pop('stride_height')
    if 'stride_width' in ARCH_SPACE:
        ARCH_SPACE.pop('stride_width')

if args.no_pooling is True:
    if 'pool_size' in ARCH_SPACE:
        ARCH_SPACE.pop('pool_size')


def parse_smi(smi_trace, gpu):
    cleared = []
    for line in smi_trace:
        line = str(line)
        index = line.find("MiB")
        if index != -1:
            cleared.append(int(line[index-5:index]))
    return(cleared[gpu])


def get_logger(filepath=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    if filepath is not None:
        file_handler = logging.FileHandler(filepath+'.log', mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(file_handler)
    return logger
    

def darts_memory(subspaces):
    fileHandler = logging.FileHandler(args.log_filename + time.strftime("%m%d_%H%M_%S",time.localtime()), mode = "a+")
    fileHandler.setLevel(logging.INFO)
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.INFO)
    logging.basicConfig(
                        handlers=[
                                    # fileHandler,
                                    streamHandler],
                        level= logging.DEBUG,
                        format='%(asctime)s %(levelname)-8s %(message)s')

    logging.info("=" * 45 + "\n" + " " * (20 + 33) + "Begin" +  " " * 20 + "\n" + " " * 33 + "=" * 45)
    
    # Find dataset. I use both windows (desktop) and Linux (server)
    # "nt" for dataset stored on windows machine and else for dataset stored on Linux
    if os.name == "nt":
        dataPath = "~/testCode/data"
    elif os.path.expanduser("~")[-5:] == "zyan2":
        dataPath = "~/Private/data/CIFAR10"
    else:
        dataPath = "/dataset/CIFAR10"
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    trainset = torchvision.datasets.CIFAR10(root=dataPath, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=dataPath, train=False, download=True, transform=transform_test)
    # Due to some interesting features of DARTS, we need two trainsets to avoid BrokenPipe Error
    # This trainset should be totally in memory, or to suffer a slow speed for num_workers=0
    # TODO: Actually DARTS uses testset here, I don't like it. This testset also needs to be in the memory anyway
    logging.debug("Caching data")
    trainset_in_memory = []
    for data in trainset:
        trainset_in_memory.append(data)
    testset_in_memory = []
    for data in testset:
        testset_in_memory.append(data)

    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize, shuffle=True)
    archLoader  = torch.utils.data.DataLoader(testset_in_memory, batch_size=args.batchSize, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize, shuffle=False, num_workers=4)

    for subspace in subspaces:
        logging.debug("Creating model")
        superModel = SuperNet()
        superModel.get_model(SubCIFARNet(subspace))
        archParams = superModel.get_arch_params()
        netParams  = superModel.get_net_params()
        # Training optimizers
        archOptimizer = optim.Adam(archParams,lr = 0.1)
        netOptimizer  = optim.Adam(netParams, lr = 1e-3)
        criterion = nn.CrossEntropyLoss()
        # GPU or CPU
        device = torch.device(args.device)
        superModel.to(device)

        best_rollout = 0
        # Train the super net
        superModel.modify_super(True)
        superModel.train_short(trainLoader, archLoader, archOptimizer, netOptimizer, criterion, device, 5)
        smi_trace = subprocess.check_output("nvidia-smi")
        memory_size = parse_smi(smi_trace, args.gpu)
        print(memory_size)


def memory(device, dir='experiment'):
    subspaces = [[[1,2],[1,0],[1,2],[1,3],[1,2],[1,3]],[[2,0],[2,1],[2,3],[2,0],[2,1],[2,0]]]
    darts_memory(subspaces)

SCRIPT = {
    'darts': memory
    # 'nested': nested_search,
    # 'quantization': quantization_search
}

def main():
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available()
                          else "cpu")
    print(f"using device {device}")
    dir = os.path.join(
        f'experiment',
        args.mode,
        'non_linear' if args.skip else 'linear',
        ('without' if args.no_stride else 'with') + ' stride, ' +
        ('without' if args.no_pooling else 'with') + ' pooling',
        args.dataset + f"({args.layers} layers)"
        )
    if os.path.exists(dir) is False:
        os.makedirs(dir)
    SCRIPT[args.mode](device, dir)

if __name__ == '__main__':
    import random
    # torch.manual_seed(args.seed)
    # random.seed(args.seed)
    main()