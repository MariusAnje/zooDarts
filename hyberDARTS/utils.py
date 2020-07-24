import numpy as np
import torch
import copy

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
        for j in range(len(space[0]) - size + 1):
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

def n_params(subspace:list, channel_size:list, linear_size:list, fm_size:list):
    conv_params = 0
    for i in range(len(subspace)):
        ks = 0
        for w in subspace[i]:
            ks += (w*2 + 1) ** 2
        # ks = (subspace[i][-1] * 2 + 1) ** 2
        conv_params += ks * channel_size[i] * channel_size[i+1]
    linear_params = 0
    for i in range(len(linear_size) - 1):
        linear_params += linear_size[i] * linear_size[i+1]
    return conv_params + linear_params

def stored_fm(subspace:list, channel_size:list, linear_size:list, fm_size:list):
    conv_fm = 0
    for i in range(len(subspace)):
        conv_fm += fm_size[i] * fm_size[i] * channel_size[i] * channel_size[i+1] * len(subspace[i])
    linear_fm = 0
    for i in range(len(linear_size) - 1):
        linear_fm += linear_size[i+1]
    return conv_fm + linear_fm

def memory_size(subspace:list):
    # a = 0.0000792633
    a = 0.000082
    # b = 1162.9887
    b = 1362.9887
    channel_size = [3, 128, 128, 256, 256, 512, 512]
    fm_size = [32, 32, 16, 16, 8, 8]
    linear_size = [512*4*4, 1024, 10]
    params  = n_params(subspace, channel_size, linear_size, fm_size)
    fm_size = stored_fm(subspace, channel_size, linear_size, fm_size)
    return a * (fm_size/10 + params) + b

def get_acc_filename(fn:str):
    """
        TODO: The name of the file recording accuracy is hardcoded
    """
    pattern = "rollout_record_"
    index = fn.find(pattern)
    pr = fn[:index]
    af = fn[index + len(pattern):]
    acc_filename = pr + "acc_" + af
    return acc_filename

def accuracy_analysis(fn:str, ep:int, th:int=4000):
    acc_filename = get_acc_filename(fn)
    rollouts = torch.load(fn)[:ep]
    acc = torch.load(acc_filename)[:ep]
    acc_sorted = np.argsort(acc)
    
    list_used = []
    for arch_index in acc_sorted:
        list_tmp = copy.deepcopy(list_used)
        list_tmp.append(rollouts[arch_index])
        WS, WSSize = working_set(list_tmp, len(list_tmp))
        subspace = []
        for item in WS:
            subspace.append(item[0])
        size = memory_size(subspace)
        if size <= th:
            list_used = copy.deepcopy(list_tmp)

    WS, WSSize = working_set(list_tmp, len(list_tmp))
    final_subspace = []
    for item in WS:
        final_subspace.append(item[0])
    return final_subspace

if __name__ == "__main__":
    import torch
    record = torch.load("rollout_record")
    record = record * 10
    subspace = min_subspace(record, 9)
    print(subspace)

