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
from darts.models import SubCIFARNet, ChildCIFARNet, QuantCIFARNet
from darts import get_data

import utils
import finetune
import time


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

def sync_search(device, dir='experiment'):
    dir = os.path.join(
        dir, f"rLut={args.rLUT}, rThroughput={args.rThroughput}")
    if os.path.exists(dir) is False:
        os.makedirs(dir)
    time_stamp = time.strftime("%m%d_%H%M%S", time.localtime())
    filepath = os.path.join(dir, f"joint_ep_{args.episodes}_{time_stamp}")
    logger = get_logger(filepath)
    csvfile = open(filepath+'.csv', mode='w+', newline='')
    writer = csv.writer(csvfile)
    logger.info(f"INFORMATION")
    logger.info(f"mode: \t\t\t\t\t {'joint'}")
    logger.info(f"dataset: \t\t\t\t {args.dataset}")
    logger.info(f"number of child network layers: \t {args.layers}")
    logger.info(f"include stride: \t\t\t {not args.no_stride}")
    logger.info(f"include pooling: \t\t\t {not args.no_pooling}")
    logger.info(f"skip connection: \t\t\t {args.skip}")
    logger.info(f"required # LUTs: \t\t\t {args.rLUT}")
    logger.info(f"required throughput: \t\t\t {args.rThroughput}")
    logger.info(f"Assumed frequency: \t\t\t {CLOCK_FREQUENCY}")
    logger.info(f"training epochs: \t\t\t {args.epochs}")
    logger.info(f"data augmentation: \t\t\t {args.augment}")
    logger.info(f"batch size: \t\t\t\t {args.batch_size}")
    logger.info(f"controller learning rate: \t\t {args.learning_rate}")
    logger.info(f"architecture episodes: \t\t\t {args.episodes}")
    logger.info(f"using multi gpus: \t\t\t {args.multi_gpu}")
    logger.info(f"architecture space: ")
    for name, value in ARCH_SPACE.items():
        logger.info(name + f": \t\t\t\t {value}")
    logger.info(f"quantization space: ")
    for name, value in QUAN_SPACE.items():
        logger.info(name + f": \t\t\t {value}")
    agent = Agent({**ARCH_SPACE, **QUAN_SPACE}, args.layers,
                  lr=args.learning_rate,
                  device=torch.device('cpu'), skip=args.skip)
    train_data, val_data = data.get_data(
        args.dataset, device, shuffle=True,
        batch_size=args.batch_size, augment=args.augment)
    input_shape, num_classes = data.get_info(args.dataset)
    writer.writerow(["ID"] +
                    ["Layer {}".format(i) for i in range(args.layers)] +
                    ["Accuracy"] +
                    ["Partition (Tn, Tm)", "Partition (#LUTs)",
                    "Partition (#cycles)", "Total LUT", "Total Throughput"] +
                    ["Time"])
    child_id, total_time = 0, 0
    logger.info('=' * 50 +
                "Start exploring architecture & quantization space" + '=' * 50)
    best_samples = BestSamples(5)
    # Bad Bad Bad Thing
    flops_channel_size = [3, 64, 64, 128, 128, 256, 256]
    flops_fm_size = [32, 32, 16, 16, 8, 8]
    flops_linear_size = [512*4*4, 1024, 10]
    # End of Bad Bad Bad Thing
    for e in range(args.episodes):
        logger.info('-' * 130)
        child_id += 1
        start = time.time()
        rollout, paras = agent.rollout()
        logger.info("Sample Architecture ID: {}, Sampled actions: {}".format(
                    child_id, rollout))

        NFlops = utils.flops(utils.RL2DR_rollout(rollout, True), flops_channel_size, flops_linear_size, flops_fm_size)
        NNFlops = np.log(NFlops) / 20.5

        arch_paras, quan_paras = utility.split_paras(paras)
        fpga_model = FPGAModel(rLUT=args.rLUT, rThroughput=args.rThroughput,
                               arch_paras=arch_paras, quan_paras=quan_paras)
        if fpga_model.validate():
            model, optimizer = child.get_model(
                input_shape, arch_paras, num_classes, device,
                multi_gpu=args.multi_gpu, do_bn=False)
            _, reward = backend.fit(
                model, optimizer, train_data, val_data, quan_paras=quan_paras,
                epochs=args.epochs, verbosity=args.verbosity)
        else:
            reward = 0

        print(f"acc: {reward}, Flops: {NFlops}")
        reward = reward + 1 - (NFlops / 4e9)

        agent.store_rollout(rollout, reward)
        end = time.time()
        ep_time = end - start
        total_time += ep_time
        best_samples.register(child_id, rollout, reward)
        writer.writerow(
            [child_id] +
            [str(paras[i]) for i in range(args.layers)] +
            [reward] + list(fpga_model.get_info()) + [ep_time]
            )
        logger.info(f"Reward: {reward}, " +
                    f"Elasped time: {ep_time}, " +
                    f"Average time: {total_time/(e+1)}")
        logger.info(f"Best Reward: {best_samples.reward_list[0]}, " +
                    f"ID: {best_samples.id_list[0]}, " +
                    f"Rollout: {best_samples.rollout_list[0]}")
    logger.info(
        '=' * 50 +
        "Architecture & quantization sapce exploration finished" +
        '=' * 50)
    logger.info(f"Total elasped time: {total_time}")
    logger.info(f"Best samples: {best_samples}")
    csvfile.close()

def nas(device, dir='experiment'):
    filepath = os.path.join(dir, f"nas ({args.episodes} episodes)")
    logger = get_logger(filepath)
    csvfile = open(filepath+'.csv', mode='w+', newline='')
    writer = csv.writer(csvfile)
    logger.info(f"INFORMATION")
    logger.info(f"mode: \t\t\t\t\t {'nas'}")
    logger.info(f"dataset: \t\t\t\t {args.dataset}")
    logger.info(f"number of child network layers: \t {args.layers}")
    logger.info(f"include stride: \t\t\t {not args.no_stride}")
    logger.info(f"include pooling: \t\t\t {not args.no_pooling}")
    logger.info(f"skip connection: \t\t\t {args.skip}")
    logger.info(f"training epochs: \t\t\t {args.epochs}")
    logger.info(f"data augmentation: \t\t\t {args.augment}")
    logger.info(f"batch size: \t\t\t\t {args.batch_size}")
    logger.info(f"controller learning rate: \t\t {args.learning_rate}")
    logger.info(f"architecture episodes: \t\t\t {args.episodes}")
    logger.info(f"using multi gpus: \t\t\t {args.multi_gpu}")
    logger.info(f"architecture space: ")
    for name, value in ARCH_SPACE.items():
        logger.info(name + f": \t\t\t\t {value}")
    agent = Agent(ARCH_SPACE, args.layers,
                  lr=args.learning_rate,
                  device=torch.device('cpu'), skip=args.skip)
    train_data, val_data = data.get_data(
        args.dataset, device, shuffle=True,
        batch_size=args.batch_size, augment=args.augment)
    input_shape, num_classes = data.get_info(args.dataset)
    writer.writerow(["ID"] +
                    ["Layer {}".format(i) for i in range(args.layers)] +
                    ["Accuracy", "Time"]
                    )
    arch_id, total_time = 0, 0
    logger.info('=' * 50 + "Start exploring architecture space" + '=' * 50)
    logger.info('-' * len("Start exploring architecture space"))
    best_samples = BestSamples(5)
    rollout_record = []
    for e in range(args.episodes):
        arch_id += 1
        start = time.time()
        arch_rollout, arch_paras = agent.rollout()
        rollout_record.append(arch_rollout)
        logger.info("Sample Architecture ID: {}, Sampled actions: {}".format(
                    arch_id, arch_rollout))
        model, optimizer = child.get_model(
            input_shape, arch_paras, num_classes, device,
            multi_gpu=args.multi_gpu, do_bn=True)
        _, arch_reward = backend.fit(
            model, optimizer, train_data, val_data,
            epochs=args.epochs, verbosity=args.verbosity)
        agent.store_rollout(arch_rollout, arch_reward)
        end = time.time()
        ep_time = end - start
        total_time += ep_time
        best_samples.register(arch_id, arch_rollout, arch_reward)
        writer.writerow([arch_id] +
                        [str(arch_paras[i]) for i in range(args.layers)] +
                        [arch_reward] +
                        [ep_time])
        logger.info(f"Architecture Reward: {arch_reward}, " +
                    f"Elapsed time: {ep_time}, " +
                    f"Average time: {total_time/(e+1)}")
        logger.info(f"Best Reward: {best_samples.reward_list[0]}, " +
                    f"ID: {best_samples.id_list[0]}, " +
                    f"Rollout: {best_samples.rollout_list[0]}")
        logger.info('-' * len("Start exploring architecture space"))
    logger.info(
        '=' * 50 + "Architecture space exploration finished" + '=' * 50)
    logger.info(f"Total elapsed time: {total_time}")
    logger.info(f"Best samples: {best_samples}")
    csvfile.close()
    torch.save(rollout_record, os.path.join("./experiment","rollout_record" + time.strftime("%m%d_%H%M_%S",time.localtime())))


    # rollout_record = torch.load("rollout_record")
    return rollout_record, best_samples.rollout_list[0]
    

def darts(subspace, device):
    fileHandler = logging.FileHandler(args.log_filename + time.strftime("%m%d_%H%M_%S",time.localtime()), mode = "a+")
    fileHandler.setLevel(logging.INFO)
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.INFO)
    logging.basicConfig(
                        handlers=[
                                    fileHandler,
                                    streamHandler],
                        level= logging.DEBUG,
                        format='%(asctime)s %(levelname)-8s %(message)s')

    logging.info("=" * 45 + "\n" + " " * (20 + 33) + "Begin" +  " " * 20 + "\n" + " " * 33 + "=" * 45)
    logging.info(args.ex_info + f" method:{args.method} e{args.epochs} ep{args.episodes} test{args.train_epochs} file_{args.rollout_filename} {args.method} wsSize:{args.wsSize}" )
    
    # # Find dataset. I use both windows (desktop) and Linux (server)
    # # "nt" for dataset stored on windows machine and else for dataset stored on Linux
    # if os.name == "nt":
    #     dataPath = "~/testCode/data"
    # elif os.path.expanduser("~")[-5:] == "zyan2":
    #     dataPath = "~/Private/data/CIFAR10"
    # else:
    #     dataPath = "/dataset/CIFAR10"
    
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    # trainset = torchvision.datasets.CIFAR10(root=dataPath, train=True, download=True, transform=transform_train)
    # testset = torchvision.datasets.CIFAR10(root=dataPath, train=False, download=True, transform=transform_test)
    # # Due to some interesting features of DARTS, we need two trainsets to avoid BrokenPipe Error
    # # This trainset should be totally in memory, or to suffer a slow speed for num_workers=0
    # # TODO: Actually DARTS uses testset here, I don't like it. This testset also needs to be in the memory anyway
    # logging.debug("Caching data")
    # trainset_in_memory = []
    # for data in trainset:
    #     trainset_in_memory.append(data)
    # testset_in_memory = []
    # for data in testset:
    #     testset_in_memory.append(data)

    # trainLoader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize, shuffle=True)
    # archLoader  = torch.utils.data.DataLoader(testset_in_memory, batch_size=args.batchSize, shuffle=True)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize, shuffle=False, num_workers=4)
    trainLoader, testloader = get_data.get_normal_loader(args.batchSize)
    archLoader = get_data.get_arch_loader(args.batchSize)

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
    # device = torch.device(args.device)
    superModel.to(device)

    # Warm up. Well, DK if warm up is needed
    # For latency, I would think of two methods. Warming only weights is one
    # Warming also arch without latency is even more interesting
    for i in range(0):
        superModel.modify_super(True)
        superModel.warm(trainLoader, netOptimizer, criterion, device)
        logging.debug(f"           arch: {superModel.get_arch_params()}")

    debug = False
    best_acc = 0
    best_rollout = 0
    for i in range(args.train_epochs):
        # Train the super net
        superModel.modify_super(True)
        superModel.train(trainLoader, archLoader, archOptimizer, netOptimizer, criterion, device)
        superAcc = superModel.test(testloader, device)
        # Test the chosen modules
        superModel.modify_super(False)
        acc = superModel.test(testloader, device)
        logging.info(f"epoch {i:-3d}:  acc: {acc:.4f}, super: {superAcc:.4f}")
        logging.info(f"           arch: {superModel.get_module_choice()}")
        logging.debug(superModel.get_arch_params())
        # record the best rollout
        if acc > best_acc:
            best_acc = acc
            best_rollout, choice_names = superModel.get_module_choice()
        torch.save(superModel.model.state_dict(), "checkpoint.pt")
    return best_rollout

def q_darts(subspace, device):
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
    logging.info(args.ex_info + f" method:{args.method} e{args.epochs} ep{args.episodes} test{args.train_epochs} file_{args.rollout_filename} {args.method} wsSize:{args.wsSize}" )

    logging.info(f"{subspace}")
    
    trainLoader, testloader = get_data.get_normal_loader(args.batchSize)
    archLoader = get_data.get_arch_loader(args.batchSize)

    logging.debug("Creating model")
    superModel = SuperNet()
    
    superModel.get_model(QuantCIFARNet(subspace))
    archParams = superModel.get_arch_params()
    netParams  = superModel.get_net_params()
    
    # Training optimizers
    archOptimizer = optim.Adam(archParams,lr = 0.1)
    netOptimizer  = optim.SGD(netParams, lr = 1e-4, momentum=0.9)
    # netOptimizer  = optim.Adam(netParams, lr = 1e-3)
    criterion = nn.CrossEntropyLoss()
    # GPU or CPU
    # device = torch.device(args.device)
    superModel.to(device)

    # Warm up. Well, DK if warm up is needed
    # For latency, I would think of two methods. Warming only weights is one
    # Warming also arch without latency is even more interesting
    for i in range(0):
        superModel.modify_super(True)
        superModel.warm(trainLoader, netOptimizer, criterion, device)
        logging.debug(f"           arch: {superModel.get_arch_params()}")

    debug = False
    best_acc = 0
    best_rollout = 0
    for i in range(args.train_epochs):
        # Train the super net
        superModel.modify_super(True)
        superModel.train(trainLoader, archLoader, archOptimizer, netOptimizer, criterion, device)
        superAcc = superModel.test(testloader, device)
        # Test the chosen modules
        superModel.modify_super(False)
        acc = superModel.test(testloader, device)
        logging.info(f"epoch {i:-3d}:  acc: {acc:.4f}, super: {superAcc:.4f}")
        logging.info(f"           arch: {superModel.get_module_choice()}")
        logging.debug(superModel.get_arch_params())
        # record the best rollout
        if acc > best_acc:
            best_acc = acc
            best_rollout, choice_names = superModel.get_module_choice()
        torch.save(superModel.model.state_dict(), "checkpoint.pt")
    return best_rollout


def ruleAll(device, dir='experiment'):
    rollout_record, dr_rollout = nas(device, dir)
    rl_rollout = utils.RL2DR_rollout(dr_rollout)
    subspace = utils.min_subspace(rollout_record, args.wsSize, args.method)
    print(subspace)
    dr_rollout = darts(subspace, device)
    print("RL best arch acc: ", finetune.main(device, rl_rollout, 60, args))
    print("DR best arch acc: ", finetune.main(device, dr_rollout, 60, args))

def darts_only(device, dir='experiment'):
    rollout_record = torch.load(args.rollout_filename)[:args.episodes]
    subspace = utils.min_subspace(rollout_record, args.wsSize, args.method)
    print(subspace)
    dr_rollout = darts(subspace, device)
    print("DR best arch acc: ", finetune.main(device, dr_rollout, 60, args))

def quant_darts_only(device, dir='experiment'):
    # rollout_record = torch.load(args.rollout_filename)[:args.episodes]
    # subspace = utils.min_subspace(rollout_record, args.wsSize, args.method)
    # print(subspace)
    # >>>> badthing here

    # >>>> RLDR_S.o1309374, e 30
    # subspace = [[0, 1], [(1, 2), (0, 1), (0, 2)], [(1, 0), (0, 0), (0, 2)], [0, 2, 3], [(1, 2), (0, 2)], [(0, 1), (1, 1), (0, 2)], [1, 2], [(1, 2), (1, 0)], [(1, 1), (0, 2)], [1, 2, 3], [(1, 0), (0, 0), (0, 2)], [(1, 2), (0, 1)], [0, 1], [(1, 0), (1, 1), (0, 2)], [(1, 2), (0, 2)], [1, 2], [(0, 1), (1, 1)], [(1, 2), (1, 1), (0, 2)]] # ep 50, size 3 --> 7409 MB
    # subspace = [[0, 1], [(1, 2), (1, 1), (0, 1), (0, 2)], [(0, 1), (1, 0), (0, 0), (1, 1)], [0, 3], [(1, 2), (0, 0), (0, 2)], [(0, 1), (1, 1), (0, 2)], [0, 1, 2, 3], [(1, 2), (1, 0), (1, 1)], [(0, 1), (1, 1), (0, 2)], [1, 2, 3], [(0, 1), (1, 0), (0, 2)], [(1, 2), (0, 2)], [1, 3], [(1, 0), (1, 1)], [(1, 2), (0, 1)], [1, 2, 3], [(0, 1), (1, 0), (1, 1)], [(1, 2), (1, 1)]] # ep 40, size 4 --> 11045 MB
    # subspace = [[0, 1], [(1, 2), (0, 1), (0, 2)], [(0, 1), (1, 0), (0, 0)], [0, 3], [(0, 0), (0, 2)], [(1, 1), (0, 2)], [0, 1, 2], [(1, 2), (1, 0)], [(0, 1), (1, 1), (0, 2)], [2, 3], [(0, 1), (1, 0), (0, 2)], [(1, 2), (0, 2)], [1], [(1, 0), (1, 1)], [(1, 2), (0, 1)], [1, 2, 3], [(0, 1), (1, 0), (1, 1)], [(1, 2), (1, 1)]] # ep 30, size 3 --> 7219 MB
    # subspace = [[0, 1], [(1, 2), (0, 1), (0, 2)], [(0, 1), (1, 0), (0, 0)], [0, 3], [(0, 0), (0, 2)], [(1, 1), (0, 2)], [0, 1, 2], [(1, 2), (1, 0)], [(0, 1), (1, 1), (0, 2)], [2, 3], [(0, 1), (1, 0), (0, 2)], [(1, 2), (0, 2)], [1], [(1, 0), (1, 1)], [(1, 2), (0, 1)], [1, 2, 3], [(0, 1), (1, 0), (1, 1)], [(1, 2), (1, 1)]] # ep 20, size 3 --> 7219 MB
    # subspace = [[0, 1], [(1, 2), (0, 0), (0, 2)], [(0, 1), (1, 0)], [0, 1, 3], [(1, 0), (0, 0), (0, 2)], [(1, 0), (1, 1), (0, 2)], [0, 1, 3], [(1, 2), (1, 0), (0, 1)], [(0, 1), (0, 2)], [1, 2], [(0, 1), (1, 0)], [(1, 2), (0, 2)], [0, 1], [(1, 0), (1, 1)], [(1, 2), (0, 1), (1, 1)], [1, 2, 3], [(0, 1), (1, 0), (0, 0)], [(1, 2), (0, 2), (1, 1)]] # ep 10, size 3 --> 9163 MB

    # >>>> RLDR_S.o1309386, e 30
    # ep 50 The same as ep 40
    # subspace = [[0, 1, 2], [(0, 0), (0, 2)], [(1, 2), (0, 0), (1, 1)], [0, 1, 2], [(1, 2), (0, 0), (0, 2)], [(1, 2), (0, 0)], [0, 3], [(0, 1), (1, 0)], [(0, 1), (1, 0), (1, 1)], [0, 1, 2], [(0, 2), (0, 0), (1, 1)], [(0, 2)], [0, 1, 3], [(1, 2), (1, 0), (1, 1)], [(1, 2), (1, 0), (1, 1)], [0, 3], [(1, 2), (1, 0), (0, 2)], [(1, 2), (0, 2)]] # ep 40, size 4 --> 8327 MB
    # subspace = [[0, 2, 3], [(1, 0), (0, 0), (0, 2)], [(1, 2), (0, 0), (0, 1)], [1, 2], [(1, 2), (0, 0), (0, 1)], [(1, 2), (1, 1)], [0, 1, 3], [(0, 1), (1, 0), (1, 2)], [(0, 1), (1, 1)], [0, 1, 2], [(0, 2), (0, 0), (1, 1)], [(1, 2), (0, 2)], [0, 3], [(0, 2), (1, 0), (1, 1)], [(1, 2), (1, 0), (0, 2)], [2, 3], [(1, 2), (0, 0), (0, 2)], [(1, 2)]] # ep 30, size 4 --> 9201 MB
    # subspace = [[0, 2, 3], [(1, 0), (0, 0), (0, 2)], [(1, 2), (0, 0), (0, 1)], [1, 2], [(1, 2), (0, 0), (0, 1)], [(1, 2), (1, 1)], [0, 1, 3], [(0, 1), (1, 0), (1, 2)], [(0, 1), (1, 1)], [0, 1, 2], [(0, 2), (0, 0), (1, 1)], [(1, 2), (0, 2)], [0, 3], [(0, 2), (1, 0), (1, 1)], [(1, 2), (1, 0), (0, 2)], [2, 3], [(1, 2), (0, 0), (0, 2)], [(1, 2)]] # ep 20, size 4 --> 9201 MB
    # subspace = [[0, 3], [(1, 0), (0, 0), (0, 2)], [(0, 1), (0, 0)], [1], [(0, 1), (0, 2), (1, 2)], [(0, 1), (0, 2), (1, 1)], [1], [(1, 2), (1, 1)], [(1, 2), (0, 0), (1, 1)], [1, 2, 3], [(1, 1), (0, 2)], [(1, 2), (0, 2), (1, 1)], [3], [(0, 1), (1, 0), (0, 2)], [(1, 0), (0, 2)], [1, 2], [(0, 0), (0, 2)], [(1, 2), (0, 0)]] # ep 10, size 4 --> 5921 MB

    # >>> RLDR_S.o1309385, e 10
    # subspace = [[0, 1], [(1, 2), (1, 0)], [(0, 1), (0, 2)], [1, 2], [(1, 2), (1, 0), (0, 2)], [(1, 2), (0, 2), (1, 1)], [1, 2], [(1, 2), (0, 0), (1, 1)], [(0, 1), (1, 1), (1, 2)], [0, 2, 3], [(1, 1), (0, 2), (0, 0)], [(1, 2), (1, 1)], [2, 3], [(1, 2), (0, 2)], [(1, 2), (0, 1), (0, 2)], [0, 2, 3], [(1, 0), (1, 1), (0, 2)], [(0, 1), (1, 2)]] # ep 50, size 3 --> 8453 MB
    # subspace = [[0, 1], [(1, 2), (1, 0)], [(0, 1), (0, 2)], [1, 2], [(1, 2), (1, 0), (0, 2)], [(1, 2), (0, 2), (1, 1)], [1, 2], [(1, 2), (0, 0), (1, 1)], [(0, 1), (1, 1), (1, 2)], [0, 2, 3], [(1, 1), (0, 2), (0, 0)], [(1, 2), (1, 1)], [2, 3], [(1, 2), (0, 2)], [(1, 2), (0, 1), (0, 2)], [0, 2, 3], [(1, 0), (1, 1), (0, 2)], [(0, 1), (1, 2)]] # ep 40, size 3 --> 8453 MB
    # subspace = [[0, 1, 2], [(1, 2), (1, 0), (1, 1), (0, 2)], [(0, 1), (0, 2)], [1, 2], [(1, 2), (1, 0), (0, 1), (0, 2)], [(1, 2), (0, 1), (0, 2)], [2, 3], [(1, 2), (0, 0), (1, 1)], [(0, 1), (1, 1), (1, 2)], [0, 3], [(1, 2), (0, 0), (0, 2)], [(1, 2), (0, 2), (1, 1)], [0, 1, 2, 3], [(1, 2), (1, 0)], [(0, 1), (1, 1), (0, 2)], [0, 2, 3], [(1, 0), (0, 2)], [(0, 1), (1, 1), (1, 2)]] # ep 30, size 4 --> 10955 MB
    # subspace = [[1, 2, 3], [(1, 2), (1, 0), (0, 2)], [(0, 1), (0, 2)], [1, 2], [(1, 2), (1, 0), (0, 1)], [(0, 1), (0, 2), (1, 2)], [0, 2], [(0, 2), (0, 0), (1, 1)], [(1, 2), (0, 1), (1, 1)], [3], [(1, 2), (0, 0), (0, 2)], [(1, 2), (1, 1), (0, 2)], [0, 2], [(1, 2), (1, 0), (0, 1)], [(0, 1)], [0, 2, 3], [(1, 0), (1, 1)], [(1, 2), (1, 1)]] # ep 20, size 3 --> 7981 MB
    # subspace = [[0, 2, 3], [(1, 2), (0, 0), (0, 2)], [(0, 1), (0, 2)], [2, 3], [(0, 1), (1, 0), (1, 1), (1, 2)], [(0, 1), (1, 1), (1, 2)], [0, 2, 3], [(0, 2), (1, 1)], [(0, 1), (1, 1)], [2, 3], [(1, 2), (0, 1), (0, 2)], [(1, 2), (1, 1), (0, 2)], [0, 2], [(0, 1), (1, 0), (0, 0), (0, 2)], [(0, 1), (1, 1)], [0, 1, 2, 3], [(1, 2), (1, 0), (1, 1)], [(1, 2), (0, 1), (1, 1)]] # ep 10, size 4 --> 10669 MB

    # >>> RLDR_S.o1309388, e 10
    # subspace = [[0, 3], [(1, 2), (0, 1), (1, 1)], [(1, 2), (0, 1), (1, 1)], [0, 2], [(1, 2), (0, 1), (1, 1)], [(0, 1), (0, 0), (1, 1)], [0, 1, 3], [(1, 2), (1, 0), (0, 0), (1, 1)], [(1, 2), (0, 2)], [0, 2, 3], [(1, 2), (1, 1)], [(1, 2), (0, 1), (0, 2)], [1, 2], [(0, 1), (1, 0), (0, 0), (0, 2)], [(0, 1), (0, 2)], [1, 3], [(1, 2), (1, 0), (1, 1), (0, 2)], [(1, 2), (0, 2), (1, 1)]] # ep 50, size 4 --> 9981 MB
    # subspace = [[0, 2, 3], [(1, 2), (0, 1), (1, 1)], [(1, 2), (0, 1), (1, 1)], [2, 3], [(0, 1), (0, 0), (1, 1)], [(0, 1), (0, 2), (1, 1)], [0, 1, 3], [(1, 2), (1, 0), (0, 0), (1, 1)], [(1, 2), (0, 1), (0, 2)], [0, 2, 3], [(1, 2), (0, 1)], [(1, 2), (0, 1), (1, 1)], [1, 2], [(0, 1), (1, 0), (0, 0), (0, 2)], [(0, 2)], [0, 3], [(0, 1), (1, 0), (1, 1), (0, 2)], [(1, 2), (0, 2), (1, 1)]] # ep 40, size 4 --> 10711 MB
    # subspace = [[0, 2, 3], [(1, 2), (0, 1), (1, 1)], [(1, 2), (0, 1), (1, 1)], [2, 3], [(0, 1), (0, 0), (1, 1)], [(0, 1), (0, 2), (1, 1)], [0, 1, 3], [(1, 2), (1, 0), (0, 0), (1, 1)], [(1, 2), (0, 1), (0, 2)], [0, 2, 3], [(1, 2), (0, 1)], [(1, 2), (0, 1), (1, 1)], [1, 2], [(0, 1), (1, 0), (0, 0), (0, 2)], [(0, 2)], [0, 3], [(0, 1), (1, 0), (1, 1), (0, 2)], [(1, 2), (0, 2), (1, 1)]] # ep 30, size 4 --> 10711 MB
    # subspace = [[1, 2, 3], [(1, 2), (0, 1)], [(1, 2), (1, 1)], [0, 2, 3], [(0, 1), (0, 0)], [(0, 1), (1, 1), (0, 2)], [0, 1, 3], [(1, 2), (1, 0), (1, 1)], [(0, 1), (0, 2), (1, 2)], [2], [(1, 2), (0, 1)], [(1, 2), (1, 0), (1, 1)], [0, 1, 2], [(0, 1), (1, 0), (1, 2)], [(0, 2)], [0, 3], [(0, 1), (1, 0), (1, 1)], [(1, 2), (1, 1)]] # ep 20, size 3 --> 7779 MB
    # subspace = [[1, 2, 3], [(1, 2), (0, 1)], [(1, 2), (1, 1)], [0, 2, 3], [(0, 1), (0, 0)], [(0, 1), (1, 1), (0, 2)], [0, 1, 3], [(1, 2), (1, 0), (1, 1)], [(0, 1), (0, 2), (1, 2)], [2], [(1, 2), (0, 1)], [(1, 2), (1, 0), (1, 1)], [0, 1, 2], [(0, 1), (1, 0), (1, 2)], [(0, 2)], [0, 3], [(0, 1), (1, 0), (1, 1)], [(1, 2), (1, 1)]] # ep 10, size 3 --> 7779 MB

    # >>> RLDR_S.o1309387, e 10
    subspace = [[1, 2, 3], [(1, 0), (0, 0), (1, 1)], [(0, 1), (1, 1), (0, 2)], [0, 1, 2], [(1, 2), (1, 0), (0, 2)], [(1, 2), (1, 1)], [2, 3], [(1, 2), (1, 0), (0, 0), (0, 2)], [(1, 2), (1, 1)], [1, 3], [(1, 0), (1, 1), (0, 2)], [(1, 2), (0, 2), (1, 1)], [0, 2, 3], [(0, 1), (1, 0), (0, 0)], [(1, 2), (0, 2)], [0, 1], [(1, 2), (0, 0), (0, 2), (1, 1)], [(1, 2), (0, 1), (1, 1)]] # ep 50, size 4 --> 9473 MB
    # ep 40, same as ep 50 
    # ep 30, same as ep 50 
    # subspace = [[1, 2, 3], [(1, 0), (0, 0), (1, 1)], [(1, 1), (0, 2)], [0, 1, 3], [(1, 2), (1, 0), (0, 2)], [(1, 2), (1, 1)], [0, 2, 3], [(1, 2), (0, 0), (0, 2)], [(1, 2), (1, 1)], [0, 1, 3], [(1, 0), (1, 1), (0, 2)], [(1, 2), (0, 0), (0, 2)], [2, 3], [(0, 1), (1, 0), (0, 0)], [(1, 2), (1, 1), (0, 2)], [0, 1], [(1, 2), (1, 1), (0, 2)], [(1, 2), (0, 2), (0, 1), (1, 1)]] # ep 20, size 4 --> 9311 MB
    # subspace = [[1, 2, 3], [(1, 2), (0, 0), (1, 1)], [(1, 0), (1, 1)], [1, 2, 3], [(1, 0), (0, 0), (0, 2)], [(1, 2), (1, 1)], [0, 2], [(1, 0), (0, 0)], [(1, 2), (1, 0), (1, 1)], [0, 3], [(1, 1), (0, 2)], [(1, 0), (0, 0), (0, 2), (1, 1)], [0, 3], [(0, 0)], [(1, 2), (0, 0), (1, 1)], [0, 1], [(1, 0), (0, 0), (1, 1)], [(0, 1), (1, 0), (0, 2)]] # ep 10, size 4 --> 7481 MB

    # >>> RLDR_S.o1309375
    # subspace = [[1, 2], [(1, 2), (0, 0)], [(1, 2), (1, 1)], [0, 1, 3], [(0, 1), (1, 0), (0, 2)], [(1, 2), (0, 2)], [1, 2, 3], [(1, 0), (1, 1)], [(1, 2), (0, 2)], [1, 2, 3], [(1, 2), (0, 1), (1, 1)], [(1, 2), (0, 1), (1, 1)], [0, 1, 2, 3], [(1, 2), (1, 0), (1, 1)], [(1, 2), (0, 2)], [0, 2, 3], [(1, 2), (1, 0), (1, 1)], [(0, 1), (0, 2), (1, 2)]] # ep 50, size 4 --> 9527 MB
    # ep 40, same as ep 50
    # subspace = [[1, 2], [(1, 2), (0, 0)], [(1, 2), (0, 2), (1, 1)], [0, 1, 3], [(0, 1), (1, 0), (0, 2), (1, 2)], [(1, 2), (0, 2)], [1, 2, 3], [(1, 0), (1, 1)], [(1, 2), (1, 1), (0, 2)], [0, 1, 2, 3], [(1, 2), (0, 0), (1, 1)], [(1, 2), (0, 0), (0, 1)], [0, 1, 2], [(1, 2), (1, 0), (1, 1)], [(1, 2), (0, 2)], [2, 3], [(1, 2), (1, 0)], [(0, 1), (1, 1), (0, 2), (1, 2)]] # ep 30, size 4 --> 10671 MB
    # subspace = [[1, 3], [(1, 2), (0, 0), (0, 1)], [(1, 2), (0, 2)], [0, 1, 3], [(1, 2), (1, 0)], [(1, 2), (0, 2)], [1, 3], [(1, 2), (1, 0)], [(0, 1), (1, 1), (0, 2)], [0, 2, 3], [(1, 2), (0, 0), (0, 2)], [(1, 2), (0, 0), (0, 2)], [0, 1], [(0, 2), (1, 0), (1, 1)], [(0, 2)], [0, 2], [(1, 2), (1, 0), (0, 2)], [(0, 1), (1, 1), (1, 2)]] # ep 20, size 3 --> 7051 MB
    # subspace = [[0, 1, 2], [(1, 2), (0, 0), (0, 2)], [(0, 1), (1, 1)], [0, 2], [(0, 1), (1, 0), (1, 1), (0, 2)], [(0, 1), (1, 1)], [0, 1, 2], [(0, 1), (1, 0), (0, 0)], [(0, 1), (1, 1), (0, 2)], [1, 2, 3], [(1, 2), (0, 2), (1, 1)], [(1, 2), (1, 0), (0, 0)], [1, 3], [(0, 1), (0, 2), (1, 2)], [(0, 1), (1, 1)], [0, 1, 3], [(0, 1), (1, 0), (1, 1)], [(0, 0), (0, 2), (1, 1)]] # ep 10, size 4 --> 10537 MB

    # >>> RLDR_S.o1309376
    # subspace = [[2], [(0, 2), (1, 0), (1, 1)], [(1, 2), (0, 2)], [1, 2, 3], [(1, 2), (1, 1)], [(0, 1), (1, 1), (1, 2)], [0, 1, 2], [(1, 2), (1, 1), (0, 2)], [(0, 1), (0, 2), (1, 1)], [0, 1, 2], [(1, 2), (1, 0), (1, 1)], [(0, 1), (0, 2), (1, 1)], [0, 2], [(0, 1), (1, 1), (1, 2)], [(1, 2), (1, 0), (0, 2)], [0, 1, 2], [(0, 0), (1, 1)], [(1, 2), (0, 2)]] # ep 50, size 4 --> 7841 MB
    # ep 40, same as ep 50
    # subspace = [[0, 2], [(1, 0), (0, 0), (1, 1)], [(1, 2), (0, 1), (0, 2)], [0, 1, 2, 3], [(1, 2), (1, 0), (1, 1)], [(0, 1), (1, 1), (1, 2)], [0, 1, 2], [(1, 2), (0, 2), (1, 1)], [(0, 1), (0, 2)], [0, 1, 2], [(1, 2), (1, 0), (1, 1)], [(0, 1), (0, 2), (1, 1)], [0, 2], [(0, 1), (0, 2), (1, 2)], [(1, 2), (0, 2)], [0, 1], [(0, 0), (1, 1)], [(1, 2), (0, 2)]] # ep 30, size 4 --> 9541 MB
    # subspace = [[2], [(1, 0), (1, 1)], [(1, 2), (0, 2)], [1, 2, 3], [(1, 2), (1, 1)], [(0, 1), (1, 1), (1, 2)], [0, 1, 2], [(1, 2), (1, 1)], [(0, 1), (0, 2)], [0, 1, 2], [(1, 0), (1, 1)], [(0, 1), (0, 2), (1, 1)], [0, 2], [(0, 1), (1, 2)], [(1, 2), (0, 2)], [0, 1], [(0, 0), (1, 1)], [(1, 2), (0, 2)]] # ep 20, size 3 --> --> 5441 MB
    # # ep 10, same as ep 20

    # # >>> RLDR_S.o1309380
    subspace = [[0, 2, 3], [(0, 2), (1, 0), (1, 1)], [(1, 2), (1, 1)], [0, 1], [(0, 1), (0, 2)], [(1, 0), (1, 1)], [0, 2, 3], [(0, 1), (0, 2), (1, 2)], [(0, 1), (0, 2), (1, 1)], [0, 1, 3], [(0, 0), (1, 1)], [(0, 1), (0, 2), (1, 2)], [1, 2, 3], [(1, 2), (1, 0), (0, 0), (0, 2)], [(1, 2), (0, 2), (0, 1), (1, 1)], [0, 1], [(1, 2), (0, 0), (1, 1)], [(0, 0), (0, 2)]] # ep 50, size 4 --> 9159 MB
    # subspace = [[0, 2, 3], [(0, 2), (1, 0), (1, 1)], [(1, 2), (1, 1)], [0, 1], [(0, 1), (0, 0), (0, 2)], [(1, 0), (1, 1)], [0, 3], [(0, 1), (0, 2), (1, 2)], [(0, 1), (0, 2)], [0, 1, 3], [(0, 1), (0, 0), (1, 1)], [(0, 1), (0, 2), (1, 2)], [1, 2, 3], [(1, 0), (0, 0), (0, 2)], [(0, 1), (0, 2), (1, 1), (1, 2)], [0, 3], [(1, 2), (1, 0), (1, 1)], [(0, 1), (0, 0), (0, 2)]] # ep 40, size 4 --> 9829 MB
    # subspace = [[0, 2, 3], [(0, 1), (1, 0), (1, 2)], [(1, 2), (0, 2), (1, 1)], [0, 1], [(0, 1), (0, 0), (1, 1)], [(0, 1), (1, 0), (0, 2)], [0, 2, 3], [(1, 2), (0, 1), (1, 1)], [(0, 1), (0, 0), (0, 2)], [1, 3], [(0, 1), (0, 0), (1, 1)], [(1, 2), (1, 0), (0, 1)], [3], [(1, 0), (0, 0)], [(1, 2), (1, 1), (0, 2)], [0, 3], [(1, 2), (1, 0), (1, 1)], [(0, 1), (0, 0), (0, 2)]] # ep 30, size 4 --> 10079 MB
    # subspace = [[0, 1, 3], [(0, 1), (1, 0), (1, 2)], [(1, 2), (1, 1)], [0, 1, 3], [(0, 1), (1, 0), (0, 0), (1, 1)], [(0, 1), (1, 0)], [0, 2, 3], [(1, 2), (0, 1)], [(0, 1), (0, 0), (0, 2)], [1, 2, 3], [(0, 1), (0, 0), (1, 1)], [(1, 2), (0, 0), (0, 1)], [3], [(0, 1), (1, 0), (0, 0)], [(1, 2), (1, 1), (0, 2)], [0, 3], [(1, 2), (1, 0)], [(0, 1), (0, 0), (1, 1)]] # ep 20, size 4 --> 10207 MB
    # subspace = [[1, 3], [(1, 2), (1, 0), (1, 1)], [(1, 2), (0, 0), (1, 1)], [0, 2, 3], [(0, 1), (1, 0), (1, 1)], [(0, 1), (1, 0)], [0, 2, 3], [(1, 2), (1, 1)], [(0, 1), (0, 2), (1, 2)], [1, 2, 3], [(1, 0), (0, 0), (1, 1)], [(1, 2), (0, 0), (1, 1)], [3], [(0, 1), (0, 0)], [(1, 2), (0, 2)], [0, 2], [(1, 2), (1, 0)], [(0, 0), (1, 1)]] # ep 10, size 3 --> 8477 MB

    # # >>> RLDR_S.o1309381
    # subspace = [[0, 3], [(0, 1), (1, 0), (1, 2)], [(0, 1), (1, 0), (0, 0)], [0], [(1, 0), (0, 0)], [(0, 1), (1, 0), (1, 1)], [0, 3], [(1, 2), (0, 0)], [(1, 2), (0, 1), (1, 1)], [1, 2, 3], [(0, 1), (1, 0), (1, 1)], [(1, 2), (1, 1)], [0, 1, 2], [(0, 1), (0, 0)], [(1, 2), (0, 2)], [0, 1, 3], [(1, 2), (1, 1)], [(1, 2), (0, 2)]] # ep 50, size 3 --> 7061 MB
    # subspace = [[0, 2], [(0, 1), (0, 2), (1, 1)], [(1, 0), (1, 1), (0, 2)], [0], [(0, 0), (1, 1)], [(1, 0), (0, 0)], [0, 3], [(0, 1), (0, 0), (1, 2)], [(1, 2), (0, 2)], [1, 2, 3], [(0, 1), (0, 0), (1, 1)], [(1, 2), (0, 2), (1, 1)], [0, 2], [(0, 0), (1, 1)], [(1, 2), (0, 2)], [0, 1], [(1, 0), (0, 0), (1, 1)], [(1, 2), (1, 1), (0, 2)]] # ep 40, size 3 --> 6517 MB
    # # ep 30, same as ep 40
    # subspace = [[0, 1, 3], [(1, 2), (1, 0), (0, 2)], [(1, 2), (0, 2), (1, 1)], [0, 2], [(0, 1), (0, 0), (1, 1)], [(1, 0), (0, 0), (0, 2)], [0, 3], [(1, 2), (0, 2)], [(1, 2), (1, 0), (1, 1)], [2, 3], [(0, 1), (1, 1)], [(1, 2), (0, 2)], [0, 1], [(0, 0)], [(1, 2), (0, 1)], [0, 2, 3], [(1, 0), (0, 0), (0, 2)], [(0, 1), (1, 1)]] # ep 20, size 3 --> 8583 MB
    # subspace = [[1, 2, 3], [(1, 2), (0, 0), (1, 1)], [(0, 1), (1, 0), (0, 2)], [0, 1], [(0, 1), (0, 0), (1, 1)], [(1, 0), (0, 0), (1, 1)], [0, 1], [(1, 2), (1, 1), (0, 2)], [(1, 0), (1, 1), (0, 2)], [2, 3], [(0, 1), (1, 1)], [(1, 2), (0, 1), (0, 2)], [1, 2, 3], [(0, 1), (0, 0), (0, 2)], [(0, 1), (0, 2)], [2, 3], [(1, 2), (0, 0), (0, 1)], [(0, 1), (1, 1)]] # ep 10, size 3 --> 9907 MB

    # # >>> RLDR_S.o1309382
    # subspace = [[0, 1, 2], [(0, 1), (1, 1)], [(1, 2), (0, 0), (1, 1)], [0, 2, 3], [(0, 2), (1, 1)], [(0, 1), (1, 1), (1, 2)], [0, 2], [(1, 2), (1, 0), (0, 2)], [(1, 2), (0, 1), (1, 1)], [1, 3], [(0, 1), (1, 1), (1, 2)], [(1, 1), (0, 2)], [0, 1], [(0, 1), (1, 0), (0, 0)], [(0, 1), (1, 0), (0, 0), (0, 2)], [2, 3], [(1, 2), (0, 2), (1, 1)], [(1, 2), (1, 1), (0, 2)]] # ep 50, size 4 --> 9695 MB
    # subspace = [[0, 2], [(0, 1), (1, 1)], [(1, 2), (0, 0), (1, 1)], [0, 3], [(0, 2), (1, 1)], [(0, 1), (1, 1), (1, 2)], [0, 2], [(1, 2), (1, 0), (0, 2)], [(1, 2), (1, 1)], [1, 3], [(0, 1), (1, 1), (1, 2)], [(0, 2)], [0, 1], [(0, 1), (1, 0), (0, 0)], [(0, 1), (1, 0), (0, 2)], [2, 3], [(1, 2), (1, 1)], [(1, 2), (0, 2)]] # ep 40, size 3 --> 6541 MB
    # subspace = [[0, 2], [(0, 1), (0, 0)], [(1, 2), (0, 0), (1, 1)], [0, 3], [(0, 2), (1, 0), (1, 1)], [(1, 2), (1, 1)], [0, 1, 2], [(1, 2), (1, 0), (0, 2)], [(1, 2), (1, 0), (1, 1)], [1, 3], [(1, 2), (1, 1)], [(0, 1), (0, 2)], [0, 3], [(1, 0), (0, 0), (0, 2)], [(1, 0), (0, 2)], [0, 2, 3], [(1, 2), (0, 2)], [(1, 2), (0, 2)]] # ep 30, size 3 --> 7449 MB 
    # subspace = [[0, 3], [(0, 1), (0, 0), (1, 1)], [(1, 2), (0, 0), (0, 1)], [0, 2, 3], [(1, 0), (0, 2)], [(1, 2), (1, 1)], [0, 1, 2], [(0, 1), (1, 0), (0, 2)], [(1, 2), (1, 0), (1, 1)], [1, 3], [(1, 2), (1, 1)], [(0, 1), (0, 2)], [0, 2, 3], [(1, 0), (0, 2)], [(1, 0), (0, 0), (0, 2)], [0, 2], [(1, 2), (0, 2)], [(1, 2), (0, 0), (0, 2)]] # ep 20, size 3 --> 7751 MB
    # subspace = [[0, 1, 3], [(0, 1), (0, 0), (1, 1)], [(1, 2), (0, 0), (0, 1)], [2, 3], [(1, 0), (1, 1), (0, 2)], [(1, 2), (0, 2), (1, 1)], [0, 1, 2], [(0, 1), (1, 0)], [(1, 0), (1, 1)], [3], [(1, 2), (1, 0), (1, 1)], [(0, 1), (0, 2)], [2, 3], [(1, 1), (0, 2)], [(0, 1), (0, 0), (0, 2)], [0, 3], [(1, 2), (1, 0), (0, 2)], [(1, 2), (0, 0), (0, 2)]] # ep 10, size 3 --> 8953 MB

    # # >>> RLDR_S.o1309383
    # subspace = [[0, 1, 3], [(1, 0), (0, 0), (0, 2)], [(1, 0), (0, 0), (0, 2), (1, 1)], [0, 2], [(0, 1), (0, 0), (1, 1)], [(1, 2), (0, 2), (0, 1), (1, 1)], [1, 2], [(0, 1), (1, 0), (1, 1)], [(0, 1), (1, 1), (0, 2)], [0, 1, 3], [(0, 1), (0, 0), (0, 2)], [(1, 2), (0, 1)], [1, 2, 3], [(1, 2), (1, 0), (0, 0), (1, 1)], [(1, 2), (0, 1)], [0, 1, 2], [(0, 1), (0, 2), (1, 0), (1, 1)], [(1, 2), (1, 1)]] # ep 50, size 4 --> 11145 MB
    # # ep 40, same as ep 50
    # # ep 30, same as ep 50
    # subspace = [[0, 3], [(1, 0), (0, 0)], [(1, 0), (0, 0), (1, 1)], [0, 2], [(0, 0), (1, 1)], [(1, 2), (0, 1), (1, 1)], [2], [(0, 1), (1, 1)], [(0, 1), (1, 1), (0, 2)], [0, 1], [(0, 1), (0, 0)], [(1, 2), (0, 1)], [1, 2, 3], [(1, 2), (1, 0), (1, 1)], [(1, 2), (0, 1)], [1, 2], [(0, 1), (1, 0), (0, 2)], [(1, 2), (1, 1)]] # ep 20, size 3 --> 6275 MB
    # subspace = [[0, 2, 3], [(1, 2), (0, 0), (0, 1)], [(1, 2), (1, 0), (0, 0), (0, 1)], [0, 2, 3], [(1, 0), (0, 0), (0, 2)], [(0, 1), (1, 0), (0, 2)], [0, 2, 3], [(0, 2), (1, 1)], [(0, 1), (1, 1), (1, 2)], [0, 1, 2], [(0, 1), (0, 0)], [(1, 2), (0, 1)], [1, 3], [(1, 2), (1, 0), (1, 1)], [(0, 1), (1, 0), (0, 0), (1, 1)], [1, 2], [(0, 1), (1, 0), (0, 0)], [(1, 2), (0, 1)]] # ep 10, size 4 --> 11029

    # # >>> RLDR_S.o1309384
    # subspace = [[0, 1], [(0, 2), (1, 0), (1, 1)], [(1, 2), (1, 1), (0, 2)], [0, 2], [(0, 1), (1, 0), (0, 0)], [(0, 2), (1, 0), (1, 1)], [0, 1, 3], [(1, 2), (0, 2), (1, 1)], [(0, 1), (0, 2), (1, 1)], [1, 3], [(1, 2), (0, 0), (1, 1)], [(1, 2), (0, 2), (1, 1)], [0, 1, 2], [(1, 2), (1, 0), (0, 2)], [(0, 1), (1, 1)], [0, 1, 2], [(1, 2), (1, 0)], [(0, 1), (0, 2), (1, 1)]] # ep 50, size 4 --> 8715 MB
    # subspace = [[0, 1], [(0, 2), (1, 0), (1, 1)], [(1, 1), (0, 2), (0, 0)], [0, 2, 3], [(1, 0), (0, 0), (1, 1)], [(1, 2), (1, 0), (0, 2)], [0, 3], [(1, 2), (1, 1), (0, 2)], [(0, 1), (0, 2)], [0, 1, 3], [(0, 0), (1, 1)], [(1, 2), (0, 2), (0, 0), (1, 1)], [0, 1, 2], [(1, 0), (0, 2)], [(0, 1), (1, 1)], [0, 1, 2, 3], [(1, 2), (0, 0)], [(0, 1), (0, 2), (1, 1), (1, 2)]] # ep 40, size 4 --> 10389 MB
    # subspace = [[0, 1], [(0, 1), (0, 2), (1, 1)], [(0, 1), (1, 1), (0, 2)], [0, 2, 3], [(0, 1), (1, 0), (0, 0)], [(0, 1), (1, 0), (0, 2)], [0, 3], [(1, 2), (0, 0), (0, 2)], [(1, 1), (0, 2)], [0, 1, 2, 3], [(0, 0), (1, 1)], [(1, 2), (0, 2)], [1, 2, 3], [(0, 1), (1, 0), (0, 2)], [(0, 2), (1, 1)], [0, 2, 3], [(1, 2), (0, 1), (0, 2)], [(0, 2), (1, 1)]] # ep 30, size 4 --> 10125 MB
    # subspace = [[0, 3], [(0, 1), (0, 0)], [(1, 0), (1, 1)], [0, 2, 3], [(1, 2), (0, 0), (1, 1)], [(0, 1), (0, 2), (1, 2)], [0, 1, 3], [(0, 0), (0, 2), (1, 1)], [(0, 1), (1, 1)], [0, 2], [(0, 1), (0, 0), (0, 2)], [(0, 1), (1, 0), (0, 2)], [1, 2], [(0, 1), (1, 0), (0, 2)], [(0, 1), (1, 1), (1, 2)], [0, 2], [(0, 1), (0, 2)], [(1, 2), (1, 0), (0, 2)]] # ep 20, size 3 --> 8047 MB
    # ep 10, same as ep 20

    # >>> RLDR_S_RL.o1333048 Fine 40
    subspace = [[1, 2], [(0, 1), (1, 0), (0, 0)], [(1, 2), (1, 1)], [0], [(0, 1), (1, 0), (0, 2)], [(0, 1), (0, 0), (1, 1)], [0, 1, 3], [(1, 2), (1, 0), (1, 1)], [(1, 2), (0, 2)], [1, 2], [(0, 1), (0, 2), (1, 2)], [(0, 1), (0, 2), (1, 1)], [1, 2, 3], [(1, 2), (0, 2)], [(1, 2), (0, 2)], [0, 3], [(1, 2), (1, 0), (0, 2)], [(0, 1), (1, 1), (0, 2)]] # ep 50, size 3 -->  7419 MB
    # ep 40, same as ep 50
    # ep 30, same as ep 40
    subspace = [[1, 2], [(0, 1), (1, 0), (0, 0)], [(1, 2), (1, 1)], [0], [(0, 1), (1, 0), (1, 1), (0, 2)], [(0, 1), (1, 0), (0, 0), (1, 1)], [0, 1, 3], [(1, 2), (1, 0), (1, 1)], [(1, 2), (0, 2)], [0, 1, 2], [(0, 1), (0, 0), (0, 2), (1, 2)], [(0, 1), (0, 2), (1, 1), (1, 2)], [1, 2, 3], [(1, 2), (0, 2)], [(1, 2), (1, 1), (0, 2)], [0, 3], [(1, 2), (1, 0), (0, 1), (0, 2)], [(0, 1), (1, 1), (0, 2), (1, 2)]] # ep 20, size 4 --> 11115 MB
    subspace = [[0, 2], [(1, 0), (0, 0)], [(1, 2), (1, 1)], [0, 2], [(1, 0), (1, 1), (0, 2)], [(0, 2), (0, 0), (1, 1)], [0, 1], [(1, 2), (1, 0)], [(1, 2), (0, 2)], [1, 2], [(0, 1), (1, 1), (1, 2)], [(0, 1), (0, 2)], [1, 2, 3], [(1, 2), (0, 1)], [(1, 2), (0, 2)], [0, 1, 3], [(1, 0), (0, 0), (0, 2)], [(0, 1), (1, 0), (1, 1)]] # ep 10, size 3 --> 7445 MB

    # >>> RLDR_S_RL.o1333048 Fine 100 the same as Fine 40

    # >>> RLDR_S_RL.o1333047 Fine 40
    subspace = [[0, 1, 2], [(1, 2), (0, 1), (1, 1)], [(1, 2), (1, 0)], [0, 1, 3], [(0, 1), (1, 0)], [(1, 2), (0, 2), (1, 1)], [0, 3], [(1, 2), (1, 1)], [(0, 2)], [3], [(1, 2), (1, 0), (1, 1)], [(1, 2), (0, 2)], [0, 1, 3], [(0, 1), (0, 0), (1, 1)], [(1, 2), (0, 2)], [0, 2, 3], [(1, 2), (0, 1), (0, 2)], [(1, 2), (1, 1)]] # ep 50, size 3 --> 7721 MB
    # ep 40, same as ep 50
    # subspace = [[0, 1], [(0, 2), (0, 0), (1, 1)], [(0, 1), (1, 0), (1, 1)], [1, 2, 3], [(1, 0), (0, 0), (1, 1)], [(1, 1), (0, 2)], [0, 3], [(0, 2), (0, 0), (1, 1)], [(1, 2), (1, 1), (0, 2)], [2, 3], [(1, 0), (1, 1)], [(1, 2), (1, 1), (0, 2)], [0, 1, 3], [(1, 2), (1, 0), (0, 0)], [(1, 2), (0, 0), (0, 2)], [0, 1, 2], [(0, 1), (0, 0), (0, 2)], [(0, 1), (1, 1)]] # ep 30, size 3 --> 8789 MB
    # subspace = [[0, 1], [(1, 2), (0, 2), (0, 0), (1, 1)], [(0, 1), (1, 0), (1, 1)], [1, 2, 3], [(1, 0), (0, 0), (1, 1)], [(1, 1), (0, 2)], [0, 3], [(0, 2), (0, 0), (1, 1)], [(1, 2), (1, 1), (0, 2)], [2, 3], [(0, 2), (1, 0), (1, 1)], [(1, 2), (1, 1), (0, 2)], [0, 1, 3], [(1, 2), (1, 0), (0, 0), (0, 2)], [(1, 2), (0, 0), (0, 2), (1, 1)], [0, 1, 2], [(0, 1), (0, 0), (0, 2), (1, 1)], [(0, 1), (1, 1)]] # ep 20, size 4 --> 10885 MB
    # subspace = [[0, 1, 2], [(0, 1), (0, 0), (1, 1)], [(1, 0), (0, 0)], [0, 1], [(0, 1), (1, 0), (1, 1), (1, 2)], [(1, 0), (1, 1), (0, 2)], [0, 1, 3], [(0, 2), (1, 1)], [(0, 1), (1, 0), (0, 2)], [2, 3], [(1, 2), (1, 0), (1, 1)], [(1, 2), (1, 0), (1, 1)], [1, 2, 3], [(0, 0), (1, 1)], [(0, 1), (0, 2)], [0, 2, 3], [(0, 1), (0, 2)], [(1, 2), (0, 2), (1, 1)]] # ep 10, size 4 -->

    # RLDR_S.o1309374, e 30, before
    subspace = [[1], [(1, 1), (0, 2)], [(0, 1), (1, 1), (0, 2)], [0, 2, 3], [(1, 2), (1, 0), (0, 0)], [(0, 1), (1, 1), (1, 2)], [0, 2, 3], [(1, 2), (0, 2), (1, 0), (1, 1)], [(0, 1), (1, 1), (0, 2)], [1, 2], [(0, 1), (0, 0), (1, 2)], [(0, 1), (0, 2)], [0, 1, 3], [(0, 1), (1, 0), (1, 1), (0, 2)], [(0, 1), (1, 1), (0, 2)], [1, 3], [(0, 1), (1, 0), (1, 1)], [(1, 2), (1, 1), (0, 2), (0, 0)]] # ep 50, size 4 --> 11013 MB
    # subspace = [[1, 2], [(1, 2), (1, 1), (0, 2)], [(0, 1), (1, 1)], [0, 2, 3], [(1, 2), (1, 0), (0, 0)], [(0, 1), (0, 2), (1, 1), (1, 2)], [0, 2, 3], [(1, 2), (0, 2), (1, 1)], [(0, 1), (0, 2), (1, 1)], [0, 1, 2], [(0, 1), (1, 0), (1, 2)], [(0, 1), (0, 2)], [1, 3], [(0, 1), (1, 0), (1, 1)], [(0, 1), (0, 2), (1, 1)], [1, 3], [(1, 0), (1, 1)], [(1, 2), (1, 0), (0, 0), (1, 1)]] # ep 40, size 4 -->  10807 MB
    # subspace = [[0, 1], [(0, 1), (0, 2), (1, 2)], [(0, 1), (1, 0), (0, 0), (1, 2)], [0, 1, 3], [(1, 0), (0, 0), (0, 2)], [(0, 2), (1, 1)], [0, 1, 2], [(1, 2), (1, 0)], [(0, 1), (0, 2), (1, 1), (1, 2)], [0, 2, 3], [(0, 1), (1, 0), (0, 2)], [(1, 2), (0, 2)], [1, 3], [(0, 2), (1, 0), (1, 1)], [(0, 1), (0, 0), (1, 2)], [1, 2, 3], [(0, 1), (1, 0), (1, 1)], [(1, 2), (0, 0), (1, 1)]] # ep 30, size 4 --> 10989 MB
    # ep 20, same as ep 30
    # subspace = [[0, 1], [(1, 2), (0, 1), (0, 2)], [(0, 1), (1, 0), (1, 2)], [0, 1, 3], [(1, 0), (0, 0), (0, 2)], [(0, 2), (1, 1)], [0, 1, 2], [(1, 2), (1, 0)], [(0, 1), (0, 2), (1, 2)], [0, 2], [(0, 1), (1, 0), (0, 2)], [(1, 2), (0, 2)], [1, 3], [(0, 2), (1, 0), (1, 1)], [(0, 1), (0, 0), (1, 2)], [1, 2, 3], [(0, 1), (1, 0)], [(1, 2), (0, 0), (1, 1)]] # ep 10, size 3 -->9321 MB

    # RLDR_S.o1309375, e 30, before
    # subspace = [[0, 1, 2], [(1, 2), (0, 0), (0, 2)], [(1, 2), (1, 1)], [0, 1, 3], [(1, 0), (0, 2)], [(1, 2), (0, 1), (0, 2)], [0, 1, 3], [(0, 1), (1, 0), (1, 1)], [(1, 2), (0, 2)], [1, 2, 3], [(1, 2), (0, 1), (1, 1)], [(0, 1), (1, 1), (1, 2)], [0, 1, 3], [(1, 2), (1, 0), (0, 1), (1, 1)], [(1, 2), (0, 1), (0, 2)], [0, 2, 3], [(1, 2), (1, 0), (0, 1), (1, 1)], [(1, 2), (0, 0), (0, 1)]] # ep 50, size 4 --> 11131 MB
    # ep 40, same as ep 50
    # subspace = [[0, 1, 2], [(1, 2), (0, 0), (0, 2)], [(1, 2), (1, 1)], [0, 1], [(1, 0), (0, 2)], [(0, 1), (0, 2)], [0, 1, 3], [(0, 1), (1, 0)], [(1, 2), (0, 2)], [1, 2], [(1, 2), (1, 1)], [(0, 1), (1, 2)], [0, 1, 3], [(1, 2), (0, 1), (1, 1)], [(1, 2), (0, 1), (0, 2)], [0, 2, 3], [(1, 2), (1, 0), (0, 1)], [(1, 2), (0, 0), (0, 1)]] # ep 30, size 3 --> 8065 MB
    # subspace = [[0, 1, 3], [(1, 2), (0, 1), (0, 2)], [(1, 2), (0, 2), (1, 1)], [0, 1], [(1, 0), (0, 2)], [(0, 1), (0, 2)], [0, 1, 3], [(0, 1), (1, 0), (1, 2)], [(0, 1), (0, 2)], [1, 2, 3], [(1, 2), (0, 2), (1, 1)], [(1, 2), (0, 2)], [0, 1, 3], [(0, 1), (0, 2), (1, 1)], [(0, 1), (0, 2)], [0, 2], [(0, 1), (1, 0), (0, 2)], [(0, 1), (0, 0), (1, 2)]] # ep 20, size 3 --> 8665 MB
    # subspace = [[0, 1, 2], [(1, 2), (0, 0), (0, 2)], [(0, 1), (1, 1)], [0, 2], [(0, 1), (1, 0), (1, 1), (0, 2)], [(0, 1), (1, 1)], [0, 1, 2], [(0, 1), (1, 0), (0, 0)], [(0, 1), (1, 1), (0, 2)], [1, 2, 3], [(1, 2), (0, 2), (1, 1)], [(1, 2), (1, 0), (0, 0)], [1, 3], [(0, 1), (0, 2), (1, 2)], [(0, 1), (1, 1)], [0, 1, 3], [(0, 1), (1, 0), (1, 1)], [(0, 0), (0, 2), (1, 1)]] # ep 10, size 4 --> 10527 MB

    # RLDR_S.o1309376, e 30, before
    # subspace = [[1, 2], [(1, 2), (1, 0), (1, 1)], [(1, 2), (0, 1), (0, 2)], [1, 2, 3], [(1, 2), (1, 1)], [(0, 1), (1, 1), (1, 2)], [0, 1, 2], [(1, 2), (0, 1), (1, 1)], [(0, 1), (1, 1), (0, 2)], [0, 1, 2], [(0, 1), (1, 0), (1, 1)], [(0, 1), (0, 2), (1, 1)], [0, 2], [(0, 1), (0, 0), (1, 2)], [(1, 2), (0, 2)], [0, 1], [(0, 0), (1, 1)], [(1, 2), (0, 2)]] # ep 50, size 4 --> 8193 MB
    # ep 40, same as ep 50
    # ep 30, same as ep 50
    # ep 20, same as ep 50
    # subspace = [[2, 3], [(0, 1), (1, 0), (1, 1)], [(1, 2), (0, 2)], [1, 2, 3], [(1, 2), (1, 1)], [(0, 1), (1, 1), (1, 2)], [0, 1, 2], [(1, 2), (1, 0), (1, 1)], [(0, 1), (0, 2)], [0, 1, 2], [(1, 2), (1, 0), (1, 1)], [(0, 1), (0, 2), (1, 1)], [0, 2], [(0, 1), (1, 2)], [(1, 2), (0, 2)], [0, 1, 3], [(0, 0), (1, 1)], [(1, 2), (0, 0), (0, 2)]] # ep 10, size 4 --> 8547 MB

    # >>>> end of badthing
    dr_rollout = q_darts(subspace, device)
    # rollout_output = utils.parse_quant_dr_rollout(subspace, dr_rollout)
    # print("DR best arch acc: ", finetune.main(device, [(rollout_output, "Normal")], 100, args, quant = True))
    rollout_output = utils.parse_quant_dr_rollout(subspace, dr_rollout, aw = True)
    print("DR best arch acc: ", finetune.main(device, [(rollout_output, "aw")], 100, args, quant = True))
    
    dr_rollout = q_darts(subspace, device)
    rollout_output = utils.parse_quant_dr_rollout(subspace, dr_rollout, aw = True)
    print("DR best arch acc: ", finetune.main(device, [(rollout_output, "aw")], 100, args, quant = True))

    dr_rollout = q_darts(subspace, device)
    rollout_output = utils.parse_quant_dr_rollout(subspace, dr_rollout, aw = True)
    print("DR best arch acc: ", finetune.main(device, [(rollout_output, "aw")], 100, args, quant = True))

SCRIPT = {
    'nas': nas,
    'all': ruleAll,
    'darts': darts_only,
    'quantization': quant_darts_only,
    'joint': sync_search
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
    print("Channels: [64,128,256]")
    SCRIPT[args.mode](device, dir)

if __name__ == '__main__':
    import random
    # torch.manual_seed(args.seed)
    # random.seed(args.seed)
    main()
