import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# tqdm is imported for better visualization
from tqdm import tqdm


class MixedBlock(nn.Module):
    def __init__(self, hardware:torch.Tensor):
        super(MixedBlock, self).__init__()
        self.hardware = hardware
        self.mix = nn.Parameter(torch.ones(len(hardware))).requires_grad_()
        self.sm = nn.Softmax(dim=-1)
        

    def forward(self):
        p = self.sm(self.mix)
        output = p.dot(self.hardware)
        return output
    def possibility(self):
        return self.sm(self.mix)
        
numHW = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
test_record = []
test_time = 100

for _ in tqdm(range(test_time)):
    test_record_item = []
    for n in tqdm(numHW, leave = False):
        a = torch.randn(n)
        a = a - a.min()
        model = MixedBlock(a)
        optimizer = optim.Adam(model.parameters(), lr=0.1)
        criterion = nn.MSELoss()
        # loader = tqdm(range(100000),leave=False, mininterval=0.1,)
        loader = range(100000)
        for i in loader:
            optimizer.zero_grad()
            output = model()
            output.backward()
            optimizer.step()
            #loader.set_description(f"{output:.4f}")
            if output < 0.01:
                # print(f"numHW = {n}, epochs = {i}")
                test_record_item.append(i)
                break
    test_record.append(test_record_item)

torch.save(test_record, "simpleLog.pt")
print(torch.load("simpleLog.pt"))