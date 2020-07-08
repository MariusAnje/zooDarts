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

def min_working_set(WS, WSSize):
    size = np.ones(len(WS))
    for j in range(len(WSSize[0])):
        for i in range(len(WSSize)):
            size[i] *= WSSize[i][j]
    index = size.argmin()
    subspace = []
    for item in WS:
        subspace.append(item[index])
    return subspace

def min_subspace(record, size):
    WS, WSSize = working_set(record, 9)
    subspace = min_working_set(WS, WSSize)
    return subspace


if __name__ == "__main__":
    import torch
    record = torch.load("rollout_record")
    record = record * 10
    subspace = min_subspace(record, 9)
    print(subspace)
