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
from darts.modules import quantize

def clear_rollouts(record, quant = True):
    if quant:
        usefulList = [0, 8, 16, 24, 32, 40]
        usefulRecord = []
        for j in usefulList:
            usefulRecord.append(record[j])
            usefulRecord += record[j+4:j+6]
            usefulRecord += record[j+6:j+8]
        return usefulRecord
    else:
        usefulList = [0, 4, 8, 12, 16, 20]
        usefulRecord = []
        for j in usefulList:
            usefulRecord.append(record[j])
        return usefulRecord

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
        inputs = quantize(inputs, 8, 8, signed=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    return running_loss / i

def test(device, loader, criterion, optimizer, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data

            images, labels = images.to(device), labels.to(device)
            images = quantize(images, 8, 8, signed=True)
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
    # print(model)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-7, weight_decay=0, amsgrad=False)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.SGD( model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    best_acc = 0
    for i in range(epochs):
        loss = train(device, trainLoader, criterion, optimizer, model)
        scheduler.step()
        acc = test(device, testloader, criterion, optimizer, model)
        print(f"loss: {loss:.4f}, acc: {acc}")
        if acc > best_acc:
            best_acc = acc
        
        if i == 5:
            if best_acc <= 0.15:
                print("Bad Arch!!!")
                return best_acc
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
    for _ in range(5):
        best_acc = execute(rollout, trainLoader, testloader, epochs, device, quant)
        if best_acc > 0.15:
            break
    return best_acc
    
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Parser User Input Arguments')
    parser.add_argument('--batchSize', action="store", type=int, default=128)
    parser.add_argument('--device', action="store", type=str, default="cuda:0")
    parser.add_argument('--epochs', action="store", type=int, default=100)
    args = parser.parse_args()
    rollout = [1,1,1,1,1,1]
    # rollout = [3,3,3,3,3,3]
    subspace = [[2, 3], [0, 1], [2], [1], [2],      [1], [0, 1], [2], [0, 1], [2],      [1, 3], [0, 1], [2], [1], [1, 2],      [2], [0, 1], [1], [1], [2],      [2], [0, 1], [2], [0, 1], [ 2],      [1, 2], [0], [1, 2], [1], [2]]
    
    subspace = [[0, 3], [0, 1], [0, 1], [1], [1], [1, 3], [0, 1], [1], [0], [1], [0], [0], [1], [1], [1], [2, 3], [1], [0, 1], [1], [1], [2, 3], [0, 1], [0, 1], [0], [1], [0, 2], [0, 1], [1], [1], [1]] # err.625 ep 80 size 2
    # rollout = [0, 0, 2, 1, 2, 0, 1, 2, 1, 2, 1, 0, 2, 1, 2, 0, 0, 1, 1, 2, 0, 1, 2, 1, 2, 1, 0, 1, 1, 2]
    # rollout = [1, 0, 2, 1, 2, 0, 1, 2, 0, 2, 1, 0, 2, 1, 2, 0, 1, 1, 1, 2, 0, 1, 2, 1, 2, 0, 0, 2, 1, 2]
    # rollout = [0, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 0, 2, 1, 2, 0, 1, 1, 1, 2, 0, 1, 2, 0, 2, 0, 0, 1, 1, 2]
    # rollout = [1, 1, 2, 1, 2, 0, 1, 2, 1, 2, 1, 1, 2, 1, 2, 0, 1, 1, 1, 2, 0, 0, 2, 1, 2, 1, 0, 2, 1, 2]
    rollout = [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1] # darts searched
    rollout = [2, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 3, 1, 0, 1, 1, 3, 1, 1, 1, 1, 1, 0, 0, 0, 1] # err.625 @ 96 --> 87.17
    
    rollout = [3, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 3, 1, 1, 1, 1, 2, 0, 1, 0, 1, 0, 0, 1, 1, 1] # err.625 @ 50 --> 84.93
    rollout = [0, 1, 1, 0, 0, 3, 0, 1, 1, 1, 0, 1, 1, 0, 1, 3, 1, 1, 1, 0, 3, 0, 1, 1, 1, 2, 1, 1, 0, 1] # darts ep 80 top 2 size 2 # 1
    rollout = [0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1] # darts ep 80 top 2 size 2 # 2
    
    rollout = [0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1] # darts ep 80 top 2 size 2 # 3
    rollout = [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 3, 1, 1, 1, 1, 3, 0, 1, 0, 0, 2, 1, 1, 0, 1] # 3412
    # rollout = [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 3, 1, 1, 1, 1, 3, 0, 0, 0, 1, 2, 0, 1, 1, 1] # 1234
    
    # rollout = [0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1] # darts ep 80 top 2 size 2 # 4
    # rollout = [0, 1, 1, 0, 1, 3, 0, 1, 1, 1, 0, 1, 1, 0, 1, 3, 1, 1, 1, 0, 3, 0, 1, 1, 1, 2, 1, 1, 1, 1] # 3412
    # rollout = [0, 0, 1, 1, 1, 3, 1, 1, 0, 1, 0, 0, 1, 1, 1, 3, 1, 0, 1, 1, 3, 1, 1, 0, 1, 2, 1, 1, 1, 1] # 1234
    
    rollout = [0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 3, 1, 1, 1, 1, 3, 0, 1, 0, 1, 0, 1, 1, 0, 1] # 12 darts ep 80 top 2 size 2 quant 2
    rollout = [0, 1, 1, 1, 0, 3, 0, 1, 0, 1, 0, 0, 1, 0, 1, 3, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1] # 12 darts ep 80 top 3 size 3 quant 2
    # rollout = [0, 1, 0, 1, 1, 3, 0, 1, 0, 1, 0, 0, 1, 0, 1, 3, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1] # 21 darts ep 80 top 3 size 3 quant 2

    # rollout = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 3, 0, 0, 0, 1, 3, 0, 1, 0, 1, 0, 0, 0, 1, 1] # err.630 rollout 177

    # rollout = [3, 0, 1, 1, 1, 
    #            1, 1, 0, 1, 1, 
    #            3, 1, 0, 1, 0, 
    #            2, 1, 1, 0, 0, 
    #            1, 0, 1, 0, 1, 
    #            1, 1, 0, 0, 0] # proxyless 1 order 52.93
    
    # rollout = [3,  1, 1, 0, 1,
    #            1,  1, 1, 1, 0,
    #            3,  1, 0, 1, 0,
    #            2,  0, 0, 1, 1,
    #            1,  0, 1, 0, 1,
    #            1,  0, 0, 1, 0, ] # proxyless 1 reverse
    
    # rollout = [3, 1, 0, 0, 1,
    #            2, 1, 0, 0, 0,
    #            3, 1, 0, 1, 1,
    #            1, 0, 1, 0, 1,
    #            0, 0, 1, 1, 1,
    #            2, 0, 0, 1, 1, ] # proxyless 2 84.13
    
    # rollout = [3,  0, 1, 1, 0,
    #            2,  0, 0, 1, 0,
    #            3,  1, 1, 1, 0,
    #            1,  0, 1, 0, 1,
    #            0,  1, 1, 0, 1,
    #            2,  1, 1, 0, 0,] # proxyless 2 reverse 10.00
    
    # rollout = [2, 1, 1, 1, 1,
    #            3, 1, 0, 1, 1,
    #            2, 1, 1, 1, 1,
    #            2, 1, 1, 1, 1,
    #            0, 1, 1, 1, 1,
    #            3, 1, 1, 1, 1,] # proxyless 3 reverse 91.36
    
    # rollout = [2, 1, 1, 1, 1,
    #            3, 1, 1, 1, 1,
    #            2, 1, 1, 1, 1,
    #            2, 1, 1, 1, 1,
    #            1, 1, 1, 1, 1,
    #            3, 1, 1, 1, 1,] # GG 90.95
    
    rollout = [1, 1, 1, 1, 1,
               1, 1, 1, 1, 1,
               1, 1, 1, 1, 1,
               1, 1, 1, 1, 1,
               1, 1, 1, 1, 1,
               1, 1, 1, 1, 1,] # best 92.23
    
    rollout = [3, 1, 1, 1, 1,
               3, 1, 1, 1, 1,
               3, 1, 1, 1, 1,
               3, 1, 1, 1, 1,
               3, 1, 1, 1, 1,
               3, 1, 1, 1, 1,] # largest 89.98

    rollout = [0, 0, 1, 1, 1, 3, 1, 2, 0, 2, 3, 0, 2, 0, 2, 2, 1, 1, 0, 1, 0, 1, 2, 1, 2, 2, 1, 1, 1, 2] # proxyless ep 145 [(1,3,5,7)(1,3)(3,6)] order 87.5
    rollout = [0, 1, 1, 0, 1, 3, 0, 2, 1, 2, 3, 0, 2, 0, 2, 2, 0, 1, 1, 1, 0, 1, 2, 1, 2, 2, 1, 2, 1, 1] # proxyless ep 145 [(1,3,5,7)(1,3)(3,6)] reverse 83.3
    rollout = [0, 0, 1, 1, 1, 3, 1, 2, 0, 2, 3, 0, 2, 0, 2, 2, 1, 1, 0, 1, 0, 1, 2, 1, 2, 2, 1, 1, 1, 2] # proxyless ep 158 [(1,3,5,7)(1,3)(3,6)] order 87.27
    rollout = [0, 1, 1, 0, 1, 3, 0, 2, 1, 2, 3, 0, 2, 0, 2, 2, 0, 1, 1, 1, 0, 1, 2, 1, 2, 2, 1, 2, 1, 1] # proxyless ep 158 [(1,3,5,7)(1,3)(3,6)] reverse 82.55
    
    rollout = [2, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 2, 0, 0, 1, 1, 3, 1, 0, 1, 1, 0, 1, 0, 1, 1] # 21 darts ep 30 top 3 size 3 quant 2 87.47
    rollout = [2, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 2, 1, 1, 0, 0, 3, 0, 1, 1, 1, 0, 0, 0, 0, 0] # 12 darts ep 30 top 3 size 3 quant 2 90.19
    rollout = [3, 1, 2, 0, 2, 0, 0, 1, 0, 2, 2, 1, 1, 0, 1, 0, 1, 0, 0, 2, 3, 1, 0, 0, 2, 1, 1, 1, 1, 2] # RL err.2181 (e10) ep 3 90.43
    rollout = [3, 0, 2, 1, 2, 0, 0, 2, 1, 2, 2, 0, 1, 1, 1, 3, 0, 2, 1, 1, 3, 0, 2, 1, 0, 2, 0, 2, 1, 2] # 12 darts ep 40 e 10 top 2 size 2 quant 2 10.00
    rollout = [3, 1, 2, 0, 2, 0, 1, 2, 0, 2, 2, 0, 1, 0, 2, 3, 1, 1, 0, 2, 3, 1, 0, 0, 2, 2, 1, 1, 1, 2] # 21 darts ep 40 e 10 top 2 size 2 quant 2 90.29

    """
        Here comes (1,3,6)
    """
    rollout = [2, 0, 1, 0, 2, 1, 1, 0, 0, 1, 2, 1, 1, 1, 1, 0, 0, 1, 1, 2, 1, 0, 0, 1, 2, 3, 1, 2, 0, 2] # RL err.2176 (e10) ep 6 87.15

    rollout = [3, 0, 1, 0, 2, 1, 1, 2, 0, 2, 3, 0, 0, 0, 2, 2, 0, 2, 1, 2, 2, 1, 0, 0, 2, 0, 0, 2, 0, 1] # RL err.2174 (e10) ep 15 84.18
    rollout = [3, 0, 0, 0, 2, 0, 0, 0, 0, 2, 3, 0, 2, 1, 1, 0, 0, 2, 1, 2, 2, 1, 0, 0, 2, 0, 1, 2, 1, 2] # 21 darts ep 30 e 10 top 2 size 2 quant 2 
    rollout = [3, 0, 2, 0, 0, 0, 0, 2, 0, 0, 3, 0, 2, 0, 0, 0, 1, 2, 0, 2, 2, 0, 2, 1, 0, 0, 0, 1, 0, 2] # 12 darts ep 30 e 10 top 2 size 2 quant 2 
    

    """
        Here comes 64 128 256
    """

    rollout = [1, 1, 2, 1, 2,
               1, 1, 2, 1, 2,
               1, 1, 2, 1, 2,
               1, 1, 2, 1, 2,
               1, 1, 2, 1, 2,
               1, 1, 2, 1, 2,] # best 91.79
    
    # rollout = [3, 1, 2, 1, 2,
    #            3, 1, 2, 1, 2,
    #            3, 1, 2, 1, 2,
    #            3, 1, 2, 1, 2,
    #            3, 1, 2, 1, 2,
    #            3, 1, 2, 1, 2,] # largest 89.91
    
    # rollout = [2, 1, 2, 0, 2, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 3, 0, 2, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1] # RL o1297379 (e10) ep 77 78.76
    rollout = [2, 1, 2, 1, 2, 3, 0, 2, 1, 2, 2, 1, 0, 1, 1, 3, 1, 1, 1, 2, 1, 1, 2, 0, 2, 3, 0, 2, 0, 1] # 21 darts ep 80 e 10 top 4 size 4 quant 2 87.87
    rollout = [2, 1, 2, 0, 2, 1, 0, 1, 1, 2, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0, 2, 0, 2, 0, 0, 2, 0, 1] # 21 darts ep 80 e 10 top 2 size 2 quant 2 85.80
    rollout = [2, 1, 2, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 3, 1, 1, 1, 1, 2, 1, 0, 0, 1, 3, 0, 2, 0, 2] # 21 darts ep 80 e 10 top 3 size 3 quant 2 88.99
    rollout = [1, 1, 1, 1, 2, 2, 0, 2, 1, 1, 2, 0, 2, 1, 1, 3, 1, 2, 0, 2, 2, 1, 0, 0, 2, 0, 1, 2, 0, 2] # 21 darts ep 80 e 10 top 4 size 4 quant 2 90.32

    rollout = [1, 0, 0, 0, 2, 2, 0, 1, 1, 2, 2, 1, 0, 1, 1, 3, 1, 0, 1, 2, 2, 1, 2, 0, 2, 0, 1, 2, 0, 2] # 12 aw darts ep 80 e 10 top 4 size 4 quant 2 87.05
    # rollout = [2, 0, 1, 1, 2, 1, 0, 2, 0, 1, 1, 0, 1, 1, 1, 3, 0, 2, 0, 2, 2, 1, 0, 0, 1, 3, 0, 1, 0, 2] # 12 aw darts ep 80 e 10 top 3 size 3 quant 2 82.28
    # rollout = [2, 0, 1, 1, 2, 1, 0, 1, 1, 2, 1, 0, 1, 1, 1, 2, 0, 2, 0, 2, 1, 0, 2, 0, 2, 0, 0, 2, 0, 1] # 12 aw darts ep 80 e 10 top 2 size 2 quant 2 77.88

    rollout = [0, 1, 2, 1, 2, 1, 0, 1, 1, 2, 2, 0, 1, 0, 2, 1, 1, 1, 1, 2, 0, 0, 2, 1, 1, 0, 1, 2, 1, 1] # o1297361 e20 ep 00 83.52
    # 21 darts ep 80 e 20 top 2 size 2 quant 2 87.62
    # 21 darts ep 80 e 20 top 3 size 3 quant 2 85.81
    # 21 darts ep 80 e 20 top 4 size 4 quant 2 84.72
    # 21 darts ep 80 e 20 top 5 size 5 quant 2 87.59

    # rollout = [1, 1, 0, 0, 2, 1, 1, 0, 1, 2, 2, 0, 1, 1, 1, 2, 1, 0, 0, 2, 1, 1, 0, 0, 1, 1, 1, 1, 1, 2] # o1297361 e30 ep 61 90.42
    # 21 darts ep 80 e 30 top 2 size 2 quant 2 90.6
    # 21 darts ep 80 e 30 top 3 size 3 quant 2 72.33
    # 21 darts ep 80 e 30 top 3 size 4 quant 2 90.51

    # >>> o1297362 e30
    rollout = [2, 1, 1, 0, 2, 1, 1, 2, 1, 1, 3, 1, 0, 1, 2, 1, 1, 1, 1, 2, 0, 1, 0, 0, 2, 3, 0, 2, 0, 1] # ep 25 82.33
    rollout = [2, 1, 1, 1, 1, 1, 0, 0, 0, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 0, 1, 2, 2, 1, 1, 0, 0] # ep 10 e 30 size 3
    rollout = [2, 0, 2, 1, 1, 1, 1, 0, 0, 2, 2, 0, 0, 0, 2, 1, 1, 1, 1, 1, 3, 1, 0, 1, 1, 2, 1, 1, 0, 0] # ep 10 e 30 size 3 aw
    # rollout = [0, 0, 2, 1, 1, 1, 0, 2, 1, 1, 2, 1, 2, 0, 2, 1, 0, 0, 1, 1, 2, 1, 0, 1, 1, 2, 1, 0, 0, 0] # ep 20 e 30 size 3
    # rollout = [0, 1, 1, 0, 2, 1, 0, 0, 0, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 0, 1, 1, 2, 0, 2, 1, 2] # ep 20 e 30 size 3 aw 87.03
    # rollout = [1, 1, 1, 1, 2, 0, 0, 2, 0, 2, 3, 1, 2, 1, 2, 1, 0, 0, 0, 2, 2, 1, 0, 0, 1, 1, 0, 2, 1, 0] # ep 40 e 30 size 4
    # rollout = [1, 0, 2, 0, 2, 0, 0, 0, 1, 1, 3, 1, 2, 1, 2, 1, 0, 0, 1, 1, 2, 1, 0, 1, 1, 1, 1, 0, 1, 2] # ep 40 e 30 size 4 aw 85.13
    # rollout = [2, 0, 2, 1, 2, 1, 0, 0, 0, 2, 3, 1, 0, 0, 1, 2, 1, 1, 1, 1, 2, 1, 2, 0, 2, 1, 0, 1, 0, 0] # ep 80 e 30 size 4
    # rollout = [2, 1, 1, 0, 2, 1, 0, 1, 0, 2, 3, 1, 1, 0, 1, 2, 1, 1, 1, 1, 2, 1, 0, 1, 1, 1, 0, 2, 0, 1] # ep 80 e 30 size 4 aw 89.53
    # 21 darts ep 120 e 30 top 3 size 4 quant 4 88.76
     
    rollout = [2, 0, 2, 1, 2, 0, 0, 2, 0, 2, 2, 1, 2, 0, 2, 1, 1, 1, 1, 2, 2, 1, 0, 0, 1, 2, 1, 0, 0, 0] # ep 05 45.09

    # >>> o1297378 e 10
    # 21 darts ep 040 e 10 size 4 quant 2 89.65
    # 21 darts ep 080 e 10 size 4 quant 2 83.83
    # 21 darts ep 120 e 10 size 4 quant 2 84.76

    rollout = [0, 0, 2, 1, 1, 2, 1, 2, 0, 2, 0, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 2, 0, 1, 3, 1, 0, 0, 1] # ep 116
    rollout = [2, 0, 1, 1, 1, 0, 1, 2, 1, 2, 1, 0, 1, 1, 2, 1, 1, 1, 0, 1, 2, 1, 2, 0, 1, 1, 0, 0, 0, 2] # ep 053
    rollout = [1, 0, 1, 1, 2, 1, 1, 1, 1, 2, 0, 0, 0, 1, 2, 0, 1, 2, 1, 2, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1] # ep 005

    # rollout = [0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 2, 0, 1, 2, 1, 1, 1, 1, 2, 0, 2, 0, 1, 0, 0, 2] # 21 darts ep 120 e 10 size 4 quant 2
    # rollout = [0, 0, 2, 0, 2, 0, 1, 1, 0, 2, 0, 1, 2, 0, 2, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1] # 21 darts ep 120 e 10 size 4 quant 2 aw
    # rollout = [0, 0, 0, 0, 1, 2, 1, 1, 0, 0, 1, 0, 1, 0, 2, 3, 1, 2, 0, 1, 2, 1, 2, 1, 1, 0, 0, 1, 1, 1] # 21 darts ep 080 e 10 size 4 quant 2
    # rollout = [0, 0, 1, 1, 1, 2, 1, 0, 0, 2, 1, 0, 0, 1, 2, 3, 0, 2, 1, 2, 2, 1, 0, 0, 1, 0, 1, 0, 0, 1] # 21 darts ep 080 e 10 size 4 quant 2 aw
    # rollout = [3, 0, 1, 1, 2, 1, 1, 2, 1, 1, 1, 0, 0, 0, 1, 0, 1, 2, 1, 0, 1, 1, 2, 0, 1, 2, 0, 1, 1, 2] # 21 darts ep 040 e 10 size 4 quant 2
    # rollout = [3, 0, 1, 1, 2, 1, 1, 2, 0, 2, 1, 1, 2, 0, 1, 0, 1, 0, 1, 2, 1, 1, 2, 0, 1, 2, 0, 1, 1, 2] # 21 darts ep 040 e 10 size 4 quant 2 aw
    # rollout = [3, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 0, 0, 1, 1, 2, 1, 0, 0, 1, 2, 0, 2, 0, 1, 2, 1, 0] # 21 darts ep 030 e 10 size 4 quant 2
    # rollout = [3, 1, 2, 1, 1, 2, 1, 2, 0, 2, 1, 0, 0, 1, 2, 1, 1, 0, 1, 2, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1] # 21 darts ep 030 e 10 size 4 quant 2 aw
    # rollout = [3, 1, 2, 1, 2, 1, 1, 1, 0, 2, 1, 1, 2, 1, 2, 0, 0, 0, 1, 1, 1, 1, 2, 1, 0, 2, 0, 1, 0, 1] # 21 darts ep 020 e 10 size 4 quant 2
    # rollout = [3, 0, 0, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 0, 1, 0, 0, 2, 1, 0, 2, 0, 1, 2, 1, 0, 1, 2] # 21 darts ep 020 e 10 size 4 quant 2 aw
    # rollout = [3, 1, 2, 1, 1, 1, 1, 2, 0, 0, 2, 1, 1, 0, 0, 1, 0, 0, 1, 1, 3, 1, 1, 0, 2, 0, 0, 1, 1, 2] # 21 darts ep 010 e 10 size 4 quant 2
    # rollout = [3, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 0, 0, 2, 3, 1, 1, 0, 2, 0, 0, 1, 1, 2] # 21 darts ep 010 e 10 size 4 quant 2 aw


    print("rollout: ", rollout)
    print("o1297378 # ep 005")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # print("Best accuracy", main(device, rollout,  10, args, True))
    print("Best accuracy", main(device, rollout, args.epochs, args, True))
