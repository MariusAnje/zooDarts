import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import getArchParams, getNetParams, archSTD
# tqdm is imported for better visualization
from tqdm import tqdm
from modules import *
import logging

def train(numEpoch, device):
    """
        Typical training scheme for offline training.
    """
    net.train()
    best_Acc = 0.0
    for epoch in range(numEpoch):  # loop over the dataset multiple times

        running_loss = 0.0
        running_std  = 0.0
        with tqdm(trainloader, leave = False) as loader:
            for i, data in enumerate(loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                std = archSTD(arch_params)
                loss = criterion(outputs, labels)
                totalLoss = std + loss 
                totalLoss.backward()
                optimizer.step()
                if i % 5 == 4: 
                    arch_opt.zero_grad()
                    searchData = next(iter(trainloader))
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
    
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward + backward + optimize
                    outputs = net(inputs)
                    std_search = archSTD(arch_params)
                    searchLoss = criterion(outputs, labels)  + std_search
                    searchLoss.backward()
                    arch_opt.step()
                    

                # print statistics
                running_loss += loss.item()
                running_std  += std.item()
                loader.set_description(f"{running_loss/(i+1):.4f}, {running_std/(i+1):.4f}")
        acc = test(device)
        super = superEval(device)
        # print(arch_params)
        logging.info(f"Epoch {epoch}: train loss: {running_loss/(i+1):.4f}, std: {running_std/(i+1):.4f}, test: {acc:.4f}, supper: {super:.4f}")
        if acc > best_Acc:
            best_Acc = acc
            torch.save(net.state_dict(), './CIFAR10.pt')

def test(device):
    """
        Typical inference scheme for both offline validation and online inference.
    """
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct/total

def superEval(device):
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net.superEval(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct/total

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler("log", mode = "a+"),
        logging.StreamHandler()
    ])
    # Determining the use scheme
    offline = True

    # Hyper parameters for training offline and inference online
    batchSize = 64
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # dataset extracting and data preprocessing
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='/dataset/CIFAR10', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root='/dataset/CIFAR10', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=4)

    # model
    # net = StupidNet()
    # print(net.state_dict().keys())
    net = SuperNet()
    # print(net.state_dict().keys())
    # exit()
    net.to(device)

    if offline:
        # Offline training

        arch_params = getArchParams(net)
        net_params  = getNetParams(net)
        # loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        # optimizer = optim.Adam(net.parameters(), lr=0.001)
        optimizer = optim.Adam(net_params, lr=0.001)
        arch_opt  = optim.Adam(arch_params, lr=0.001)

        # Training
        train(30, device)

        # Validation
        state_dict = torch.load("./CIFAR10.pt")
        net.load_state_dict(state_dict)
        logging.info(f"Test accuracy: {test(device)}")

    else:
        # Online inference

        # The pretrained model
        state_dict = torch.load("./CIFAR10.pt")
        net.load_state_dict(state_dict)

        # Actual inference
        print(f"Test accuracy: {test(device)}")
        print(f"Test accuracy: {superEval(device)}")
