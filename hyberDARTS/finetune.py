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

from darts.models import ChildCIFARNet, QuantChildCIFARNet

def train(device, loader, criterion, optimizer, model):
    model.train()
    running_loss = 0.0
    best_Acc = 0.0
    for i, data in enumerate(loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def test(device, loader, criterion, optimizer, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct/total

def execute(rollout, trainLoader, testloader, epochs, device, quant):
    if quant:
        model = QuantChildCIFARNet(rollout)
    else:
        model = ChildCIFARNet(rollout)
    print(model)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.SGD( model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    best_acc = 0
    for _ in range(epochs):
        train(device, trainLoader, criterion, optimizer, model)
        # scheduler.step()
        acc = test(device, testloader, criterion, optimizer, model)
        print(acc)
        if acc > best_acc:
            best_acc = acc
    return best_acc

def main(device, rollout, epochs, args, quant = False):
    if os.name == "nt":
        dataPath = "~/testCode/data"
    elif os.path.expanduser("~")[-5:] == "zyan2":
        dataPath = "~/Private/data/CIFAR10"
    else:
        dataPath = "~/Private/data/CIFAR10"
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    trainset = torchvision.datasets.CIFAR10(root=dataPath, train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=dataPath, train=False, download=False, transform=transform_test)


    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize, shuffle=False, num_workers=4)
    best_acc = execute(rollout, trainLoader, testloader, epochs, device, quant)
    return best_acc
    
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Parser User Input Arguments')
    parser.add_argument('--batchSize', action="store", type=int, default=128)
    parser.add_argument('--device', action="store", type=str, default="cuda:0")
    args = parser.parse_args()
    rollout = [1,1,1,1,1,1]
    # rollout = [3,3,3,3,3,3]
    subspace = [[2, 3], [0, 1], [2], [1], [2],      [1], [0, 1], [2], [0, 1], [2],      [1, 3], [0, 1], [2], [1], [1, 2],      [2], [0, 1], [1], [1], [2],      [2], [0, 1], [2], [0, 1], [ 2],      [1, 2], [0], [1, 2], [1], [2]]
    # rollout = [0, 0, 2, 1, 2, 0, 1, 2, 1, 2, 1, 0, 2, 1, 2, 0, 0, 1, 1, 2, 0, 1, 2, 1, 2, 1, 0, 1, 1, 2]
    # rollout = [1, 0, 2, 1, 2, 0, 1, 2, 0, 2, 1, 0, 2, 1, 2, 0, 1, 1, 1, 2, 0, 1, 2, 1, 2, 0, 0, 2, 1, 2]
    # rollout = [0, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 0, 2, 1, 2, 0, 1, 1, 1, 2, 0, 1, 2, 0, 2, 0, 0, 1, 1, 2]
    # rollout = [1, 1, 2, 1, 2, 0, 1, 2, 1, 2, 1, 1, 2, 1, 2, 0, 1, 1, 1, 2, 0, 0, 2, 1, 2, 1, 0, 2, 1, 2]
    rollout = [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1] # darts searched
    rollout = [2, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 3, 1, 0, 1, 1, 3, 1, 1, 1, 1, 1, 0, 0, 0, 1] # err.625 @ 96 --> 87.17
    
    # rollout = [3, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 3, 1, 1, 1, 1, 2, 0, 1, 0, 1, 0, 0, 1, 1, 1] # err.625 @ 50 --> 84.93
    print("Rollout:", rollout)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Best accuracy", main(device, rollout, 100, args, True))