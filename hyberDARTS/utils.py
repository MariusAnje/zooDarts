import numpy as np
import torch
import copy

def space_recognition(record, quant):
    if quant:
        usefulList = [0, 8, 16, 24, 32, 40]
        newList = []
        # for i in usefulList:
        #     newList.append(i)
        #     newList += list(range(i+4,i+8))
        # usefulList = newList
        usefulRecord = []
        for j in usefulList:
            recordItemOp = []
            recordItemA  = []
            recordItemW  = []
            for i in range(len(record)):
                recordItemOp.append(record[i][j])
                recordItemA.append(tuple(record[i][j+4:j+6]))
                recordItemW.append(tuple(record[i][j+6:j+8]))
            usefulRecord += [recordItemOp, recordItemA, recordItemW]
        return usefulRecord
    else:
        usefulList = [0, 4, 8, 12, 16, 20]
        usefulRecord = []
        for j in usefulList:
            recordItem = []
            for i in range(len(record)):
                recordItem.append(record[i][j])
            usefulRecord.append(recordItem)
        return usefulRecord

def working_set(record, size, quant):
    space = space_recognition(record, quant)
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

def clear_recurse(WS:list):
    cleared = []
    for item in WS:
        if len(item) > 1:
            raise Exception("Only list with one item is clearable")
        cleared.append(item[0])
    return cleared

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
    # print(size)
    subspace = []
    for item in WS:
        subspace.append(item[index])
    return subspace

def min_subspace(record, size, find_type = "mult", quant = False):
    WS, WSSize = working_set(record, size, quant)
    subspace = min_working_set(WS, WSSize, find_type)
    return subspace

def RL2DR_rollout(rl_rollout, quant = False):
    if quant:
        usefulList = [0, 8, 16, 24, 32, 40]
        dr_rollout = []
        for i in usefulList:
            dr_rollout.append(rl_rollout[i])
            dr_rollout += rl_rollout[i+4:i+8]
        return dr_rollout
    else:
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
        conv_fm += fm_size[i] * fm_size[i] * channel_size[i+1] * len(subspace[i])
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

def parse_quant_dr_rollout(subspace, rollout_record, aw = False):
    rollout = [0 for i in range(len(subspace))]

    """
        These are for quant space of 4
    """
    # slice_point = [0]
    # slice_zie   = []
    # for i in range(0,len(subspace),5):
    #     slice_zie.append(len(subspace[i]) + 1)
    # for i in range(len(slice_zie)):
    #     slice_point.append(slice_zie[i] + slice_point[i])

    # rollout_output = []
    
    # for i in range(len(slice_zie)):
    #     op_choice = rollout_record[slice_point[i]]
    #     start = i * 5
    #     w_i_s = subspace[start + 3]
    #     w_f_s = subspace[start + 4]
    #     a_i_s = subspace[start + 1]
    #     a_f_s = subspace[start + 2]
    #     layer_quant_params = []
    #     quant_keys = ['weight_num_int_bits','weight_num_frac_bits', 'act_num_int_bits', 'act_num_frac_bits']

    #     for wi in range(len(w_i_s)):
    #         for wf in range(len(w_f_s)):
    #             for ai in range(len(a_i_s)):
    #                 for af in range(len(a_f_s)):
    #                     new_quant = [w_i_s[wi], w_f_s[wf], a_i_s[ai], a_f_s[af]]
    #                     layer_quant_params.append(new_quant)
    #     quant_params = layer_quant_params[rollout_record[slice_point[i] + op_choice + 1]]
    #     rollout_output.append(subspace[start][op_choice])
    #     rollout_output += quant_params

    """
        These are for quant space of 2
    """
    
    slice_point = [0]
    slice_zie   = []
    for i in range(0,len(subspace),3):
        slice_zie.append(len(subspace[i]) + 1)
    for i in range(len(slice_zie)):
        slice_point.append(slice_zie[i] + slice_point[i])

    rollout_output = []
    if not aw:
        for i in range(len(slice_zie)):
            op_choice = rollout_record[slice_point[i]]
            start = i * 3
            a_s = subspace[start + 2]
            w_s = subspace[start + 1]
            layer_quant_params = []
            quant_keys = ['weight_num_int_bits','weight_num_frac_bits', 'act_num_int_bits', 'act_num_frac_bits']

            for w in range(len(w_s)):
                for a in range(len(a_s)):
                    new_quant = [w_s[w][0], w_s[w][1], a_s[a][0], a_s[a][1]]
                    layer_quant_params.append(new_quant)
            quant_params = layer_quant_params[rollout_record[slice_point[i] + op_choice + 1]]
            rollout_output.append(subspace[start][op_choice])
            rollout_output += quant_params
    else:
        for i in range(len(slice_zie)):
            op_choice = rollout_record[slice_point[i]]
            start = i * 3
            a_s = subspace[start + 1]
            w_s = subspace[start + 2]
            layer_quant_params = []
            quant_keys = ['weight_num_int_bits','weight_num_frac_bits', 'act_num_int_bits', 'act_num_frac_bits']

            for w in range(len(w_s)):
                for a in range(len(a_s)):
                    new_quant = [a_s[a][0], a_s[a][1], w_s[w][0], w_s[w][1]]
                    layer_quant_params.append(new_quant)
            quant_params = layer_quant_params[rollout_record[slice_point[i] + op_choice + 1]]
            rollout_output.append(subspace[start][op_choice])
            rollout_output += quant_params

    return rollout_output

if __name__ == "__main__":
    # import torch
    # record = torch.load("rollout_record")
    # record = record * 10
    # subspace = min_subspace(record, 9)
    # print(subspace)
    
    # RLDR_S.o1309380
    suffix = "ep 50"
    q_rollouts = [
        [0, 0, 0, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 2, 1, 1, 2, 0, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 2, 1, 2, 1, 0, 0, 0, 0, 0, 0, 2], # ep 042 87.14
        [2, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 3, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 2, 0, 2], # ep 033 87.03
        [2, 0, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 1, 1, 0, 2], # ep 038 85.15
        [3, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 0, 1, 1, 0, 3, 0, 0, 0, 1, 2, 0, 2, 3, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 2, 0, 0], # ep 008 84.48
        [2, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 2, 1, 2, 2, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 1, 2, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 3, 0, 0, 0, 1, 1, 1, 1], # ep 049 81.47
        [2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 2, 0, 2, 1, 0, 0, 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 1, 1, 1, 3, 0, 0, 0, 1, 2, 0, 1], # ep 041 72.12
    ]

    size = 3
    WS, WSSize = working_set(q_rollouts[:size], size, quant = True)
    subspace = clear_recurse(WS)
    print("    subspace = " + str(subspace) + f" # {suffix}, size 3 --> ")
    size = 4
    WS, WSSize = working_set(q_rollouts[:size], size, quant = True)
    subspace = clear_recurse(WS)
    print("    subspace = " + str(subspace) + f" # {suffix}, size 4 --> ")
    suffix = "ep 40"
    q_rollouts = [
        [2, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 3, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 2, 0, 2], # ep 033 87.03
        [2, 0, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 1, 1, 0, 2], # ep 038 85.15
        [3, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 0, 1, 1, 0, 3, 0, 0, 0, 1, 2, 0, 2, 3, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 2, 0, 0], # ep 008 84.48
        [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 0, 0, 0, 0, 1, 1, 2, 3, 0, 0, 0, 1, 0, 1, 2, 3, 0, 0, 0, 1, 0, 0, 1], # ep 012 68.68
        [3, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 1, 1, 3, 0, 0, 0, 1, 0, 0, 1], # ep 019 23.05
        [2, 0, 0, 0, 1, 2, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2, 2, 0, 0, 0, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 1, 1, 0, 2], # ep 023 17.19
    ]

    size = 3
    WS, WSSize = working_set(q_rollouts[:size], size, quant = True)
    subspace = clear_recurse(WS)
    print("    subspace = " + str(subspace) + f" # {suffix}, size 3 --> ")
    size = 4
    WS, WSSize = working_set(q_rollouts[:size], size, quant = True)
    subspace = clear_recurse(WS)
    print("    subspace = " + str(subspace) + f" # {suffix}, size 4 --> ")
    suffix = "ep 30"
    q_rollouts = [
        [3, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 0, 1, 1, 0, 3, 0, 0, 0, 1, 2, 0, 2, 3, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 2, 0, 0], # ep 008 84.48
        [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 0, 0, 0, 0, 1, 1, 2, 3, 0, 0, 0, 1, 0, 1, 2, 3, 0, 0, 0, 1, 0, 0, 1], # ep 012 68.68
        [3, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 1, 1, 3, 0, 0, 0, 1, 0, 0, 1], # ep 019 23.05
        [2, 0, 0, 0, 1, 2, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2, 2, 0, 0, 0, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 1, 1, 0, 2], # ep 023 17.19
        [2, 0, 0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 0, 2, 0, 1, 3, 0, 0, 0, 1, 1, 1, 1, 2, 0, 0, 0, 0, 1, 0, 1, 3, 0, 0, 0, 1, 1, 1, 0, 2, 0, 0, 0, 1, 0, 0, 1], # ep 027 16.14
        [1, 0, 0, 0, 1, 2, 1, 1, 3, 0, 0, 0, 1, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 1, 2, 0, 0, 0, 1, 1, 0, 0, 3, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0, 1, 0, 1, 1], # ep 007 14.81
    ]

    size = 3
    WS, WSSize = working_set(q_rollouts[:size], size, quant = True)
    subspace = clear_recurse(WS)
    print("    subspace = " + str(subspace) + f" # {suffix}, size 3 --> ")
    size = 4
    WS, WSSize = working_set(q_rollouts[:size], size, quant = True)
    subspace = clear_recurse(WS)
    print("    subspace = " + str(subspace) + f" # {suffix}, size 4 --> ")
    suffix = "ep 20"
    q_rollouts = [
        [3, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 0, 1, 1, 0, 3, 0, 0, 0, 1, 2, 0, 2, 3, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 2, 0, 0], # ep 008 84.48
        [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 0, 0, 0, 0, 1, 1, 2, 3, 0, 0, 0, 1, 0, 1, 2, 3, 0, 0, 0, 1, 0, 0, 1], # ep 012 68.68
        [3, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 1, 1, 3, 0, 0, 0, 1, 0, 0, 1], # ep 019 23.05
        [1, 0, 0, 0, 1, 2, 1, 1, 3, 0, 0, 0, 1, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 1, 2, 0, 0, 0, 1, 1, 0, 0, 3, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0, 1, 0, 1, 1], # ep 007 14.81
        [0, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 0, 1, 1, 0, 3, 0, 0, 0, 0, 2, 1, 2, 1, 0, 0, 0, 0, 1, 0, 1, 3, 0, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 1, 2, 1, 0], # ep 016 13.76
        [3, 0, 0, 0, 1, 1, 1, 2, 0, 0, 0, 0, 1, 0, 0, 2, 2, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 2, 1, 1, 3, 0, 0, 0, 0, 0, 0, 2], # ep 014 13.47
    ]

    size = 3
    WS, WSSize = working_set(q_rollouts[:size], size, quant = True)
    subspace = clear_recurse(WS)
    print("    subspace = " + str(subspace) + f" # {suffix}, size 3 --> ")
    size = 4
    WS, WSSize = working_set(q_rollouts[:size], size, quant = True)
    subspace = clear_recurse(WS)
    print("    subspace = " + str(subspace) + f" # {suffix}, size 4 --> ")
    suffix = "ep 10"
    q_rollouts = [
        [3, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 0, 1, 1, 0, 3, 0, 0, 0, 1, 2, 0, 2, 3, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 2, 0, 0], # ep 008 84.48
        [1, 0, 0, 0, 1, 2, 1, 1, 3, 0, 0, 0, 1, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 1, 2, 0, 0, 0, 1, 1, 0, 0, 3, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0, 1, 0, 1, 1], # ep 007 14.81
        [1, 0, 0, 0, 1, 1, 0, 0, 2, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 2, 1, 0, 0, 0, 1, 0, 1, 1, 3, 0, 0, 0, 0, 1, 0, 2, 2, 0, 0, 0, 1, 2, 1, 1], # ep 004 12.25
        [1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 3, 0, 0, 0, 0, 1, 1, 2, 1, 0, 0, 0, 1, 2, 0, 1, 2, 0, 0, 0, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 1, 1], # ep 009 11.59
        [1, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 3, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 1, 1, 3, 0, 0, 0, 0, 1, 1, 0, 3, 0, 0, 0, 0, 0, 0, 2], # ep 006 10.75
        [2, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 1, 0, 1, 0, 3, 0, 0, 0, 0, 0, 1, 1, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1], # ep 000 10.58
    ]

    size = 3
    WS, WSSize = working_set(q_rollouts[:size], size, quant = True)
    subspace = clear_recurse(WS)
    print("    subspace = " + str(subspace) + f" # {suffix}, size 3 --> ")
    size = 4
    WS, WSSize = working_set(q_rollouts[:size], size, quant = True)
    subspace = clear_recurse(WS)
    print("    subspace = " + str(subspace) + f" # {suffix}, size 4 --> ")

    # print("40")
    # subspace = [[0, 1, 2], [(0, 0), (0, 2)], [(1, 2), (0, 0), (1, 1)], [0, 1, 2], [(1, 2), (0, 0), (0, 2)], [(1, 2), (0, 0)], [0, 3], [(0, 1), (1, 0)], [(0, 1), (1, 0), (1, 1)], [0, 1, 2], [(0, 2), (0, 0), (1, 1)], [(0, 2)], [0, 1, 3], [(1, 2), (1, 0), (1, 1)], [(1, 2), (1, 0), (1, 1)], [0, 3], [(1, 2), (1, 0), (0, 2)], [(1, 2), (0, 2)]]
    # rollout_record = [0, 1, 5, 5, 1, 1, 0, 1, 1, 5, 0, 0, 2, 1, 1, 2, 1, 5, 1, 1, 0, 0, 3]
    # rollout_output = parse_quant_dr_rollout(subspace, rollout_record, False)
    # print("Normal")
    # print(rollout_output)
    # print("AW")
    # rollout_output = parse_quant_dr_rollout(subspace, rollout_record, True)
    # print(rollout_output)

    # print("30")
    # subspace = [[0, 2, 3], [(1, 0), (0, 0), (0, 2)], [(1, 2), (0, 0), (0, 1)], [1, 2], [(1, 2), (0, 0), (0, 1)], [(1, 2), (1, 1)], [0, 1, 3], [(0, 1), (1, 0), (1, 2)], [(0, 1), (1, 1)], [0, 1, 2], [(0, 2), (0, 0), (1, 1)], [(1, 2), (0, 2)], [0, 3], [(0, 2), (1, 0), (1, 1)], [(1, 2), (1, 0), (0, 2)], [2, 3], [(1, 2), (0, 0), (0, 2)], [(1, 2)]]
    # rollout_record = [0, 4, 5, 5, 1, 4, 2, 2, 1, 3, 1, 1, 2, 3, 1, 0, 6, 2, 1, 1, 0, 3]
    # rollout_output = parse_quant_dr_rollout(subspace, rollout_record, False)
    # print("Normal")
    # print(rollout_output)
    # print("AW")
    # rollout_output = parse_quant_dr_rollout(subspace, rollout_record, True)
    # print(rollout_output)
    
    # print("20")
    # subspace = [[0, 2, 3], [(1, 0), (0, 0), (0, 2)], [(1, 2), (0, 0), (0, 1)], [1, 2], [(1, 2), (0, 0), (0, 1)], [(1, 2), (1, 1)], [0, 1, 3], [(0, 1), (1, 0), (1, 2)], [(0, 1), (1, 1)], [0, 1, 2], [(0, 2), (0, 0), (1, 1)], [(1, 2), (0, 2)], [0, 3], [(0, 2), (1, 0), (1, 1)], [(1, 2), (1, 0), (0, 2)], [2, 3], [(1, 2), (0, 0), (0, 2)], [(1, 2)]]
    # rollout_record = [0, 5, 5, 5, 1, 1, 0, 1, 0, 5, 4, 0, 3, 1, 4, 1, 0, 7, 0, 0, 1, 3]
    # rollout_output = parse_quant_dr_rollout(subspace, rollout_record, False)
    # print("Normal")
    # print(rollout_output)
    # print("AW")
    # rollout_output = parse_quant_dr_rollout(subspace, rollout_record, True)
    # print(rollout_output)
    
    # print("10")
    # subspace = [[0, 3], [(1, 0), (0, 0), (0, 2)], [(0, 1), (0, 0)], [1], [(0, 1), (0, 2), (1, 2)], [(0, 1), (0, 2), (1, 1)], [1], [(1, 2), (1, 1)], [(1, 2), (0, 0), (1, 1)], [1, 2, 3], [(1, 1), (0, 2)], [(1, 2), (0, 2), (1, 1)], [3], [(0, 1), (1, 0), (0, 2)], [(1, 0), (0, 2)], [1, 2], [(0, 0), (0, 2)], [(1, 2), (0, 0)]]
    # rollout_record = [1, 1, 2, 0, 8, 0, 5, 2, 1, 1, 4, 0, 4, 1, 1, 1, 3]
    # rollout_output = parse_quant_dr_rollout(subspace, rollout_record, False)
    # print("Normal")
    # print(rollout_output)
    # print("AW")
    # rollout_output = parse_quant_dr_rollout(subspace, rollout_record, True)
    # print(rollout_output)

    # rollout_output = parse_quant_dr_rollout(subspace, rollout_record)
    # print(rollout_output)

