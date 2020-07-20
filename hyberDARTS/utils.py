import numpy as np

def space_recognition(record):
    usefulList = [0, 4, 8, 12, 16, 20]
    usefulRecord = []
    for j in usefulList:
        recordItem = []
        for i in range(len(record)):
            recordItem.append(record[i][j])
        usefulRecord.append(recordItem)
    return usefulRecord

def working_set(record, size):
    space = space_recognition(record)
    WS = []
    WSSize = []
    for i in range(len(space)):
        WSLine = []
        WSSizeLine = []
        for j in range(len(space[0]) - size):
            WSItem = list(set(space[i][j:j+size]))
            WSLine.append(WSItem)
            WSSizeLine.append(len(WSItem))
        WS.append(WSLine)
        WSSize.append(WSSizeLine)
    return WS, WSSize

def min_working_set(WS, WSSize, find_type):
    if find_type == "mult":
        size = np.ones(len(WS[0]))
        for j in range(len(WSSize[0])):
            for i in range(len(WSSize)):
                size[j] *= WSSize[i][j]
        index = size.argmin()
    elif find_type == "add":
        size = np.zeros(len(WS[0]))
        for j in range(len(WSSize[0])):
            for i in range(len(WSSize)):
                size[j] += WSSize[i][j]
        index = size.argmin()
    elif find_type == "comp":
        size = np.zeros(len(WS[0]))
        for j in range(len(WSSize[0])):
            for i in range(len(WSSize)):
                for k in range(len(WS[i][j])):
                    size[j] += (WS[i][j][k] * 2 + 1) **2
        index = size.argmin()
    print(size)
    subspace = []
    for item in WS:
        subspace.append(item[index])
    return subspace

def min_subspace(record, size, find_type = "mult"):
    WS, WSSize = working_set(record, size)
    subspace = min_working_set(WS, WSSize, find_type)
    return subspace

def RL2DR_rollout(rl_rollout):
    usefulList = [0, 4, 8, 12, 16, 20]
    dr_rollout = []
    for i in usefulList:
        dr_rollout.append(rl_rollout[i])
    return dr_rollout

def memory_size(subspace):
    pass



if __name__ == "__main__":
    import torch
    record = torch.load("rollout_record")
    record = record * 10
    subspace = min_subspace(record, 9)
    print(subspace)

