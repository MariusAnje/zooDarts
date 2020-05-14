import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# tqdm is imported for better visualization
from tqdm import tqdm

def train(numEpoch, device):
    """
        Typical training scheme for offline training.
    """
    net.train()
    for epoch in range(numEpoch):  # loop over the dataset multiple times

        running_loss = 0.0
        best_Acc = 0.0
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

class Block(nn.Module):
    # supported type: CONV1, CONV3, CONV5, CONV7, ID
    def __init__(self, bType:str, in_channels:int, out_channels:int, norm:bool = False):
        super(Block, self).__init__()
        if bType == "ID":
            self.op = nn.Identity()
        elif bType == "CONV1":
            self.op = nn.Conv2d(in_channels, out_channels, 1)
        elif bType == "CONV3":
            self.op = nn.Conv2d(in_channels, out_channels, 3, padding = 1)
        elif bType == "CONV5":
            self.op = nn.Conv2d(in_channels, out_channels, 5, padding = 2)
        elif bType == "CONV7":
            self.op = nn.Conv2d(in_channels, out_channels, 7, padding = 3)
        self.act = nn.ReLU()
        if norm:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
        
        

    def forward(self, x):

        return self.act(self.norm(self.op(x)))

class MixedBlock(nn.Module):
    def __init__(self, modules:list, in_channels, out_channels):
        super(MixedBlock, self).__init__()
        moduleList = []
        for m in modules:
            moduleList.append(Block(m, in_channels, out_channels))
        self.moduleList = nn.ModuleList(moduleList)
        self.mix = nn.Parameter(torch.Tensor(len(modules))).requires_grad_()
        self.sm = nn.Softmax(dim=0)
        

    def forward(self, x):
        p = self.sm(self.mix)
#         print(p)
        output = p[0] * self.moduleList[0](x)
        for i in range(1, len(self.moduleList)):
            output += p[i] * self.moduleList[i](x)
        return output
    
class SuperNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(SuperNet, self).__init__()
        modules = ["CONV1", "CONV3", "CONV5", "CONV7"]
        self.block1 = MixedBlock(modules, 3, 128)
        self.block2 = MixedBlock(modules, 128, 128)
        self.block3 = MixedBlock(modules, 128, 256)
        self.block4 = MixedBlock(modules, 256, 256)
        self.block5 = MixedBlock(modules, 256, 512)
        self.block6 = MixedBlock(modules, 512, 512)
        self.pool = nn.MaxPool2d(2)
        self.lastPool = nn.AdaptiveAvgPool2d((4,4))
        self.classifier = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.lastPool(x)
        x = self.classifier(x.view(-1, 512*4*4))
        return x


if __name__ == "__main__":
    # Determining the use scheme
    offline = True

    # Hyper parameters for training offline and inference online
    if offline:
        batchSize = 64
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        batchSize = 8
        device = torch.device("cpu")

    # dataset extracting and data preprocessing
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='/dataset/CIFAR10', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root='/dataset/CIFAR10', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=4)

    # model
    net = SuperNet()
    net.to(device)

    if offline:
        # Offline training

        # loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        # Training
        train(30, device)

        # Validation
        state_dict = torch.load("./CIFAR10.pt")
        net.load_state_dict(state_dict)
        print(f"Test accuracy: {test(device)}")

    else:
        # Online inference

        # The pretrained model
        state_dict = torch.load("./CIFAR10.pt")
        net.load_state_dict(state_dict)

        # Actual inference
        print(f"Test accuracy: {test(device)}")
