import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# tqdm is imported for better visualization
import tqdm
from models import SuperCIFARNet
from modules import SuperNet, MixedBlock
import logging
import os

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--batchSize', action="store", type=int, default=64)
    parser.add_argument('--pretrained', action="store_true")
    parser.add_argument('--train_epochs', action="store", type = int, default = 10)
    parser.add_argument('--log_filename', action="store", type = str, default = "log")
    parser.add_argument('--device', action="store", type = str, default = "cuda:0")
    parser.add_argument('--ex_info', action="store", type = str, default = "nothing special")
    args = parser.parse_args()

    # Initial loggings, show DEBUG level on screen and write INFO level in file
    fileHandler = logging.FileHandler(args.log_filename, mode = "a+")
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
    logging.info(args.ex_info)
    
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
    superModel.get_model(SuperCIFARNet())
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
    for i in range(5):
        superModel.modify_super(True)
        superModel.warm(trainLoader, netOptimizer, criterion, device)
        logging.debug(f"           arch: {superModel.get_arch_params()}")

    c_gradT = []
    l_gradT = []
    debug = False
    for i in range(args.train_epochs):
        # Train the super net
        superModel.modify_super(True)
        if debug:
            c_gradList, l_gradList = superModel.train_debug(trainLoader, archLoader, archOptimizer, netOptimizer, criterion, device)
            c_gradT.append(c_gradList)
            l_gradT.append(l_gradList)
        else:
            superModel.train(trainLoader, archLoader, archOptimizer, netOptimizer, criterion, device)
        superAcc = superModel.test(testloader, device)
        # Test the chosen modules
        superModel.modify_super(False)
        acc = superModel.test(testloader, device)
        logging.info(f"epoch {i:-3d}:  acc: {acc:.4f}, super: {superAcc:.4f}")
        logging.info(f"           arch: {superModel.get_module_choice()}")
        logging.debug(superModel.get_arch_params())
        torch.save(superModel.model.state_dict(), "checkpoint.pt")
    
    if debug:
        torch.save([c_gradT, l_gradT], "gradients.pt")

