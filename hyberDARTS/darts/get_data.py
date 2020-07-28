import torch
import torchvision
import os
from torchvision import transforms

def get_dataset():
    if os.name == "nt":
        dataPath = "~/testCode/data"
    elif os.path.expanduser("~")[-5:] == "zyan2" or os.path.expanduser("~")[-len("zheyuyan"):] == "zheyuyan":
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
    trainset = torchvision.datasets.CIFAR10(root=dataPath, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=dataPath, train=False, download=True, transform=transform_test)
    
    return trainset, testset

def get_set_in_memory(dataset):
    mem_set = []
    for data in dataset:
        mem_set.append(data)
    return mem_set

def get_normal_loader(batch_size):
    trainset, testset = get_dataset()
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return trainLoader, testloader

def get_arch_loader(batch_size):
    _, testset = get_dataset()
    testset_in_memory = get_set_in_memory(testset)
    archLoader  = torch.utils.data.DataLoader(testset_in_memory, batch_size=batch_size, shuffle=True)
    return archLoader

if __name__ == "__main__":
    trainLoader, testloader = get_normal_loader(16)
    print(next(iter(trainLoader)))
