import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# tqdm is imported for better visualization
from tqdm import tqdm
from utils import *
from hwModule import HWnet, HWMixedBlock
import logging
import argparse


def train(numEpoch, device):
    """
        Typical training scheme for offline training.
    """
    net.train()
    best_Acc = 0.0
    for epoch in range(numEpoch):  # loop over the dataset multiple times

        running_loss = 0.0
        with tqdm(trainloader, leave = False) as loader:
            for i, data in enumerate(loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                loader.set_description(f"{running_loss/(i+1):.4f}")
        acc = test(device)
        print(f"Epoch {epoch}: test accuracy: {acc:.4f}")
        if acc > best_Acc:
            best_Acc = acc
            torch.save(net.state_dict(), './MIX.pt')

def test(device):
    """
        Typical inference scheme for both offline validation and online inference.
    """
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct/total

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--test_only', action="store_true")
    parser.add_argument('--ori_filename', action="store", type = str, default = "./CIFAR10_ori.pt")
    parser.add_argument('--train_epochs', action="store", type = int, default = 10)
    parser.add_argument('--log_filename', action="store", type = str, default = "hwSuper_log")
    args = parser.parse_args()
    
    
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler("args.log_filename", mode = "a+"),
        logging.StreamHandler()
    ])
    # Determining the use scheme
    offline = not args.test_only

    # Hyper parameters for training offline and inference online
    batchSize = 64
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # dataset extracting and data preprocessing
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='/dataset/CIFAR10', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=8)
    testset = torchvision.datasets.CIFAR10(root='/dataset/CIFAR10', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=8)

    # model
    # net = StupidNet()
    # print(net.state_dict().keys())
    model_used = HWnet
    state_dict = transfer_from_ori(model_used, args.ori_filename)
    net = model_used()
    net.load_state_dict(state_dict)
    # print(net.state_dict().keys())
    # exit()
    net.to(device)

    if offline:
        # Offline training

        # loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        # optimizer = optim.Adam(net.parameters(), lr=0.001)
        optimizer = optim.Adam(getArchParams(net, HWMixedBlock), lr=0.001)

        logging.info(f"Before training, test accuracy: {test(device)}")
        # Training
        train(args.train_epochs, device)

        # Validation
        state_dict = torch.load("./MIX.pt")
        net.load_state_dict(state_dict)
        logging.info(f"Test accuracy: {test(device)}")

    else:
        # Online inference

        # The pretrained model
        state_dict = torch.load("./MIX.pt")
        net.load_state_dict(state_dict)

        # Actual inference
        print(f"Test accuracy: {test(device)}")