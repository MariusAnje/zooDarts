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
    

def darts(subspace):
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
            best_rollout = superModel.get_module_choice()
        torch.save(superModel.model.state_dict(), "checkpoint.pt")
    return best_rollout
    
def ruleAll(device, dir='experiment'):
    rollout_record, dr_rollout = nas(device, dir)
    rl_rollout = utils.RL2DR_rollout(dr_rollout)
    subspace = utils.min_subspace(rollout_record, args.wsSize, args.method)
    print(subspace)
    dr_rollout = darts(subspace)
    print("RL best arch acc: ", finetune.main(device, rl_rollout, 60, args))
    print("DR best arch acc: ", finetune.main(device, dr_rollout, 60, args))

def darts_only(device, dir='experiment'):
    rollout_record = torch.load(args.rollout_filename)[:args.episodes]
    subspace = utils.min_subspace(rollout_record, args.wsSize, args.method)
    print(subspace)
    dr_rollout = darts(subspace)
    print("DR best arch acc: ", finetune.main(device, dr_rollout, 60, args))

SCRIPT = {
    'nas': nas,
    'all': ruleAll,
    'darts': darts_only
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