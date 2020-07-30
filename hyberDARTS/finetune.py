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

from darts.models import ChildCIFARNet

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

def execute(rollout, trainLoader, testloader, epochs, device):
    model = ChildCIFARNet(rollout)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_acc = 0
    for _ in range(epochs):
        train(device, trainLoader, criterion, optimizer, model)
        acc = test(device, testloader, criterion, optimizer, model)
        # print(acc)
        if acc > best_acc:
            best_acc = acc
    return best_acc

def main(device, rollout, epochs, args):
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
    trainset = torchvision.datasets.CIFAR10(root=dataPath, train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=dataPath, train=False, download=False, transform=transform_test)


    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize, shuffle=False, num_workers=4)
    best_acc = execute(rollout, trainLoader, testloader, epochs, device)
    return best_acc
    
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Parser User Input Arguments')
    parser.add_argument('--batchSize', action="store", type=int, default=128)
    parser.add_argument('--device', action="store", type=str, default="cuda:0")
    args = parser.parse_args()
    rollout = [1,1,1,1,1,1]
    # rollout = [3,3,3,3,3,3]
    print("Rollout:", rollout)
    device = torch.device(args.device)
    print("Best accuracy", main(device, rollout, 60, args))