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
from darts.modules import SuperNet#, MixedBlock
from darts.models import SubCIFARNet, ChildCIFARNet, SuperCIFARNet

import utils
import finetune
import subprocess
import finetune



# def get_args():
parser = argparse.ArgumentParser('Parser User Input Arguments')
parser.add_argument(
    'mode',
    default='nas',
    choices=['nas', 'darts', 'memory', 'joint', 'nested', 'quantization'],
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
    default=1,
    help="the total epochs for model fitting, default is 30"
    )
parser.add_argument(
    '-ep', '--episodes',
    type=int,
    default=80,
    help='''the number of episodes for training the policy network, default
        is 80'''
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
parser.add_argument('--log_filename', action="store", type = str, default = "./experiment/dr_memory")
parser.add_argument('--device', action="store", type = str, default = "cuda:0")
parser.add_argument('--ex_info', action="store", type = str, default = "nothing special")
parser.add_argument('--rollout_filename', action="store", type = str, default = "./experiment/rollout")
parser.add_argument('--method', action="store", type = str, default = "comp")
parser.add_argument('--size', action="store", type = int, default = 9)
args = parser.parse_args()
if args.gpu != 0:
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

def similar(x, y):
    return abs(x - y) < 20

def parse_smi(smi_trace, gpu):
    smi_trace = smi_trace.splitlines()
    cleared = []
    for line in smi_trace:
        line = str(line)
        index = line.find("MiB")
        if index != -1:
            cleared.append(int(line[index-5:index]))
    return(cleared[gpu])

def get_memory_consumption(gpu):
    smi_trace = subprocess.check_output("nvidia-smi")
    return(parse_smi(smi_trace, gpu))

def generate_subspaces(sample_size):
    subspace_list= []
    full_space = [0,1,2,3]
    i = 0
    j = 0
    while(True):
        subspace = []
        for _ in range(6):
            subspace.append(list(np.sort(np.random.choice(full_space,np.random.randint(1,5),False))))
        if not (subspace in subspace_list):
            subspace_list.append(subspace)
            i += 1
        j += 1
        if i >= sample_size:
            break
    return subspace_list

def generate_rollouts(sample_size):
    rollout_list= []
    full_space = [0,1,2,3]
    i = 0
    j = 0
    while(True):
        rollout = []
        for _ in range(6):
            rollout.append(int(np.random.choice(full_space,1,False)))
        if not (rollout in rollout_list):
            rollout_list.append(rollout)
            i += 1
        j += 1
        if i >= sample_size:
            break
    return rollout_list


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
    

def darts_memory(subspaces, extra_info = ""):
    fileHandler = logging.FileHandler(args.log_filename + time.strftime("%m%d_%H%M_%S",time.localtime()), mode = "a+")
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
    
    gpu_list = []
    for i in range(4):
        if get_memory_consumption(i) < 20:
            gpu_list.append(i)
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
    testset_in_memory = []
    for data in testset:
        testset_in_memory.append(data)

    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize, shuffle=True)
    archLoader  = torch.utils.data.DataLoader(testset_in_memory, batch_size=args.batchSize, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize, shuffle=False, num_workers=4)
    device = torch.device(args.device)

    # for i in range(len(subspaces)):
    for the_gpu in gpu_list:
        size_record = []
        i = 0
        print(f"GPU: {the_gpu}")
        for subspace in subspaces:
            # logging.debug("Creating model")
            superModel = SuperNet()
            superModel.get_model(SubCIFARNet(subspace))
            # superModel.get_model(SuperCIFARNet())
            archParams = superModel.get_arch_params()
            netParams  = superModel.get_net_params()
            # Training optimizers
            archOptimizer = optim.Adam(archParams,lr = 0.1)
            netOptimizer  = optim.Adam(netParams, lr = 1e-3)
            criterion = nn.CrossEntropyLoss()
            torch.cuda.empty_cache()
            # GPU or CPU
            
            superModel.to(device)
            best_rollout = 0
            # Train the super net
            superModel.modify_super(True)
            superModel.train_short(trainLoader, archLoader, archOptimizer, netOptimizer, criterion, device, 5)
            # superModel.train(trainLoader, archLoader, archOptimizer, netOptimizer, criterion, device)
            # memory_size = torch.cuda.memory_cached(device)/1024/1024
            memory_size = get_memory_consumption(the_gpu)
            logging.info(f"{extra_info}: {subspace}, {memory_size}")
            size_record.append((subspace, memory_size))
            if i == 4:
                if (similar(size_record[1][1], size_record[2][1]) or similar(size_record[2][1], size_record[3][1])):
                    break
            i += 1
        if i == len(subspaces):
            break


    torch.save(size_record, "dr_memory_record_new_" + time.strftime("%m%d_%H%M_%S",time.localtime()))

def darts_memory_trace(subspaces):
    fileHandler = logging.FileHandler(args.log_filename + time.strftime("%m%d_%H%M_%S",time.localtime()), mode = "a+")
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
    testset_in_memory = []
    for data in testset:
        testset_in_memory.append(data)

    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize, shuffle=True)
    archLoader  = torch.utils.data.DataLoader(testset_in_memory, batch_size=args.batchSize, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize, shuffle=False, num_workers=4)

    size_record = []

    i = 0
    for subspace, ep, th in subspaces:
        # logging.debug("Creating model")
        superModel = SuperNet()
        superModel.get_model(SubCIFARNet(subspace))
        # superModel.get_model(SuperCIFARNet())
        archParams = superModel.get_arch_params()
        netParams  = superModel.get_net_params()
        # Training optimizers
        archOptimizer = optim.Adam(archParams,lr = 0.1)
        netOptimizer  = optim.Adam(netParams, lr = 1e-3)
        criterion = nn.CrossEntropyLoss()
        torch.cuda.empty_cache()
        # GPU or CPU
        device = torch.device(args.device)
        superModel.to(device)
        

        best_rollout = 0
        # Train the super net
        superModel.modify_super(True)
        superModel.train_short(trainLoader, archLoader, archOptimizer, netOptimizer, criterion, device, 5)
        # superModel.train(trainLoader, archLoader, archOptimizer, netOptimizer, criterion, device)
        memory_size = torch.cuda.memory_cached(device)/1024/1024
        logging.info(f"ep:{ep}, th:{th} --> {subspace}, {memory_size}")
        size_record.append((subspace, memory_size))

    # torch.save(size_record, "dr_memory_record_new_" + time.strftime("%m%d_%H%M_%S",time.localtime()))

def nas(device, dir='experiment'):
    gpu_list = []
    for i in range(4):
        if get_memory_consumption(i) < 20:
            gpu_list.append(i)
    fileHandler = logging.FileHandler(args.log_filename + time.strftime("%m%d_%H%M_%S",time.localtime()), mode = "a+")
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

    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize, shuffle=False, num_workers=4)
    
    
    rollouts = generate_rollouts(args.size)
    print(rollouts)
    device = torch.device(args.device)
    
    for the_gpu in gpu_list:
        rollout_record = []
        i = 0
        print(f"GPU: {the_gpu}")
        for rollout in rollouts:
            torch.cuda.empty_cache()
            finetune.execute(rollout, trainLoader, testloader, args.epochs, device)
            memory_size = get_memory_consumption(the_gpu)
            logging.info(f"{rollout}, mem:{memory_size}")
            rollout_record.append((rollout, memory_size))
            if i == 4:
                if (similar(rollout_record[1][1], rollout_record[2][1]) or similar(rollout_record[2][1], rollout_record[3][1])):
                    break
            i += 1
        if i == len(rollouts):
            break
    torch.save(rollout_record, os.path.join("./experiment","rollout_mem" + time.strftime("%m%d_%H%M_%S",time.localtime())))


    # rollout_record = torch.load("rollout_record")
    # return rollout_record, best_samples.rollout_list[0]

def memory(device, dir='experiment'):
    subspaces = generate_subspaces(args.size)
    # subspaces = [subspaces[3]]  * 100
    darts_memory(subspaces)

def from_trace(device, dir='experiment'):
    th_list = [2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000]
    subspaces = []
    for  ep in range(10,args.episodes+1,10):
        
        for th in th_list:
            subspace = utils.accuracy_analysis(fn = args.rollout_filename, ep = ep, th = th)
            # print((subspace, ep, th))
            subspaces.append((subspace, ep, th))
    # print(subspaces)
    darts_memory_trace(subspaces)

SCRIPT = {
    'darts':    from_trace,
    'memory':   memory,
    'nas':      nas,
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
