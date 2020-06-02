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
    args = parser.parse_args()

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

    if os.name == "nt":
        dataPath = "~/testCode/data"
    else:
        dataPath = "/dataset/CIFAR10"
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root=dataPath, train=True, download=True, transform=transform)
    logging.debug("Caching data")
    trainset_in_memory = []
    for data in trainset:
        trainset_in_memory.append(data)
    trainLoader = torch.utils.data.DataLoader(trainset_in_memory, batch_size=args.batchSize, shuffle=True)
    archLoader  = torch.utils.data.DataLoader(trainset_in_memory, batch_size=args.batchSize, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root=dataPath, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize, shuffle=False, num_workers=4)

    logging.debug("Creating model")
    superModel = SuperNet()
    superModel.get_model(SuperCIFARNet())
    archParams = superModel.get_arch_params()
    netParams  = superModel.get_net_params()
    archOptimizer = optim.Adam(archParams,lr = 0.1)
    netOptimizer  = optim.Adam(netParams, lr = 1e-3)
    criterion = nn.CrossEntropyLoss()
    device = torch.device(args.device)
    superModel.to(device)

    for i in range(0):
        superModel.modify_super(True)
        superModel.warm(trainLoader, netOptimizer, criterion, device)
        logging.debug(f"           arch: {superModel.get_arch_params()}")

    for i in range(args.train_epochs):
        superModel.modify_super(True)
        superModel.train(trainLoader, archLoader, archOptimizer, netOptimizer, criterion, device)
        superAcc = superModel.test(testloader, device)
        superModel.modify_super(False)
        acc = superModel.test(testloader, device)
        logging.info(f"epoch {i:-3d}:  acc: {acc:.4f}, super: {superAcc:.4f}")
        logging.info(f"           arch: {superModel.get_module_choice()}")
        # logging.debug(superModel.get_arch_params())
        torch.save(superModel.model.state_dict(), "checkpoint.pt")

