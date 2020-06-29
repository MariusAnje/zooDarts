import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../Interface")
sys.path.append("../../Performance_Model")
from model_modify import *
import model_modify
import random
import train
import time
import datetime
#
# layers format:
# [ ["layer4.0.conv1", "layer4.0.conv2", "layer4.0.bn1", (256, 480, 512)],
#   ...
# ]

def resnet_18_dr(pattern_idx, k_expand, ch_list, q_list, args):

    layer_names = ["layer1.0.conv1","layer1.0.conv2","layer1.1.conv1",
        "layer1.1.conv2","layer2.0.conv2","layer2.1.conv1","layer2.1.conv2"]


    if k_expand == 0:
        layer_kernel_inc = []
    elif k_expand == 1:
        layer_kernel_inc = ["layer2.0.conv1"]
    elif k_expand == 2:
        layer_kernel_inc = ["layer2.0.downsample.0"]
    else:
        layer_kernel_inc = ["layer2.0.conv1","layer2.0.downsample.0"]

    channel_cut_layers = [["layer1.0.conv1", "layer1.0.conv2", "layer1.0.bn1", (64, 64, 64)],
                          ["layer1.1.conv1", "layer1.1.conv2", "layer1.1.bn1", (64, 64, 64)],
                          ["layer2.0.conv1", "layer2.0.conv2", "layer2.0.bn1", (64, 128, 128)],
                          ["layer2.1.conv1", "layer2.1.conv2", "layer2.1.bn1", (128, ch_list[0], 128)],
                          ["layer3.0.conv1", "layer3.0.conv2", "layer3.0.bn1", (128, ch_list[1], 256)],
                          ["layer3.1.conv1", "layer3.1.conv2", "layer3.1.bn1", (256, ch_list[2], 256)],
                          ["layer4.0.conv1", "layer4.0.conv2", "layer4.0.bn1", (256, ch_list[3], 512)],
                          ["layer4.1.conv1", "layer4.1.conv2", "layer4.1.bn1", (512, ch_list[4], 512)]]

    quant_layers = ["layer3.0.conv1", "layer3.0.conv2",
                    "layer3.1.conv1", "layer3.1.conv2",
                    "layer4.0.conv1", "layer4.0.conv2",
                    "layer4.1.conv1", "layer4.1.conv2"]
    quan_paras = {}
    quan_paras["layer3.0.conv1"] = [0, q_list[0], True]
    quan_paras["layer3.0.conv2"] = [0, q_list[1], True]
    quan_paras["layer3.1.conv1"] = [0, q_list[2], True]
    quan_paras["layer3.1.conv2"] = [0, q_list[3], True]
    quan_paras["layer4.0.conv1"] = [0, q_list[4], True]
    quan_paras["layer4.0.conv2"] = [0, q_list[5], True]
    quan_paras["layer4.1.conv1"] = [0, q_list[6], True]
    quan_paras["layer4.1.conv2"] = [0, q_list[7], True]

    return layer_names, layer_kernel_inc, channel_cut_layers, quant_layers, quan_paras

def resnet_18_dr_pre_dna(model, pattern_idx, k_expand, ch_list, q_list, args):

    parttern_77_space = pattern_sets_generate_3((7, 7))
    parttern_77 = {}
    for i in parttern_77_space.keys():
        parttern_77[i] = parttern_77_space[i].reshape((7, 7))
    layer_names_77 = ["conv1"]

    pattern_space = pattern_sets_generate_3((3, 3))
    pattern = {}
    i = 0
    for idx in pattern_idx:
        pattern[i] = pattern_space[idx].reshape((3, 3))
        i+=1
    layer_names = ["layer1.0.conv1","layer1.0.conv2","layer1.1.conv1",
        "layer1.1.conv2","layer2.0.conv2","layer2.1.conv1","layer2.1.conv2"]


    if k_expand == 0:
        layer_kernel_inc = []
    elif k_expand == 1:
        layer_kernel_inc = ["layer2.0.conv1"]
    elif k_expand == 2:
        layer_kernel_inc = ["layer2.0.downsample.0"]
    else:
        layer_kernel_inc = ["layer2.0.conv1","layer2.0.downsample.0"]

    channel_cut_layers = [
                          ["layer1.0.conv1", "layer1.0.conv2", "layer1.0.bn1", (64, 64, 64)],
                          ["layer1.1.conv1", "layer1.1.conv2", "layer1.1.bn1", (64, 64, 64)],
                          ["layer2.0.conv1", "layer2.0.conv2", "layer2.0.bn1", (64, 128, 128)],
                          ["layer2.1.conv1", "layer2.1.conv2", "layer2.1.bn1", (128, ch_list[0], 128)],
                        #   ["layer3.0.conv1", "layer3.0.conv2", "layer3.0.bn1", (128, ch_list[1], 256)],
                        #   ["layer3.1.conv1", "layer3.1.conv2", "layer3.1.bn1", (256, ch_list[2], 256)],
                        #   ["layer4.0.conv1", "layer4.0.conv2", "layer4.0.bn1", (256, ch_list[3], 512)],
                        #   ["layer4.1.conv1", "layer4.1.conv2", "layer4.1.bn1", (512, ch_list[4], 512)]
                        ]

    quant_layers = [
                    "layer3.0.conv1", "layer3.0.conv2",
                    "layer3.1.conv1", "layer3.1.conv2",
                    # "layer4.0.conv1", "layer4.0.conv2",
                    # "layer4.1.conv1", "layer4.1.conv2"
                    ]
    quan_paras = {}
    quan_paras["layer3.0.conv1"] = [0, q_list[0], True]
    quan_paras["layer3.0.conv2"] = [0, q_list[1], True]
    quan_paras["layer3.1.conv1"] = [0, q_list[2], True]
    quan_paras["layer3.1.conv2"] = [0, q_list[3], True]
    quan_paras["layer4.0.conv1"] = [0, q_list[4], True]
    quan_paras["layer4.0.conv2"] = [0, q_list[5], True]
    quan_paras["layer4.1.conv1"] = [0, q_list[6], True]
    quan_paras["layer4.1.conv2"] = [0, q_list[7], True]


    model_modify.Channel_Cut(model, channel_cut_layers)
    # model_modify.Kernel_Patter(model, layer_names, pattern, args)
    model_modify.Kenel_Expand(model, layer_kernel_inc)
    model_modify.Kenel_Quantization(model, quant_layers, quan_paras)

    model_modify.Kernel_Patter(model, layer_names_77, parttern_77, args)

    return model

# [1,22,49,54], 3, [100,210,210,470,470]
def resnet_18_space(model, pattern_idx, k_expand, ch_list, q_list, args):

    parttern_77_space = pattern_sets_generate_3((7, 7))
    parttern_77 = {}
    for i in parttern_77_space.keys():
        parttern_77[i] = parttern_77_space[i].reshape((7, 7))
    layer_names_77 = ["conv1"]

    pattern_space = pattern_sets_generate_3((3, 3))
    pattern = {}
    i = 0
    for idx in pattern_idx:
        pattern[i] = pattern_space[idx].reshape((3, 3))
        i+=1
    layer_names = ["layer1.0.conv1","layer1.0.conv2","layer1.1.conv1",
        "layer1.1.conv2","layer2.0.conv2","layer2.1.conv1","layer2.1.conv2"]


    if k_expand == 0:
        layer_kernel_inc = []
    elif k_expand == 1:
        layer_kernel_inc = ["layer2.0.conv1"]
    elif k_expand == 2:
        layer_kernel_inc = ["layer2.0.downsample.0"]
    else:
        layer_kernel_inc = ["layer2.0.conv1","layer2.0.downsample.0"]

    channel_cut_layers = [["layer1.0.conv1", "layer1.0.conv2", "layer1.0.bn1", (64, 64, 64)],
                          ["layer1.1.conv1", "layer1.1.conv2", "layer1.1.bn1", (64, 64, 64)],
                          ["layer2.0.conv1", "layer2.0.conv2", "layer2.0.bn1", (64, 128, 128)],
                          ["layer2.1.conv1", "layer2.1.conv2", "layer2.1.bn1", (128, ch_list[0], 128)],
                          ["layer3.0.conv1", "layer3.0.conv2", "layer3.0.bn1", (128, ch_list[1], 256)],
                          ["layer3.1.conv1", "layer3.1.conv2", "layer3.1.bn1", (256, ch_list[2], 256)],
                          ["layer4.0.conv1", "layer4.0.conv2", "layer4.0.bn1", (256, ch_list[3], 512)],
                          ["layer4.1.conv1", "layer4.1.conv2", "layer4.1.bn1", (512, ch_list[4], 512)]]

    quant_layers = ["layer3.0.conv1", "layer3.0.conv2",
                    "layer3.1.conv1", "layer3.1.conv2",
                    "layer4.0.conv1", "layer4.0.conv2",
                    "layer4.1.conv1", "layer4.1.conv2"]
    quan_paras = {}
    quan_paras["layer3.0.conv1"] = [0, q_list[0], True]
    quan_paras["layer3.0.conv2"] = [0, q_list[1], True]
    quan_paras["layer3.1.conv1"] = [0, q_list[2], True]
    quan_paras["layer3.1.conv2"] = [0, q_list[3], True]
    quan_paras["layer4.0.conv1"] = [0, q_list[4], True]
    quan_paras["layer4.0.conv2"] = [0, q_list[5], True]
    quan_paras["layer4.1.conv1"] = [0, q_list[6], True]
    quan_paras["layer4.1.conv2"] = [0, q_list[7], True]


    model_modify.Channel_Cut(model, channel_cut_layers)
    model_modify.Kernel_Patter(model, layer_names, pattern, args)
    model_modify.Kenel_Expand(model, layer_kernel_inc)
    model_modify.Kenel_Quantization(model, quant_layers, quan_paras)

    model_modify.Kernel_Patter(model, layer_names_77, parttern_77, args)

    return model


def get_space():

    space_name = ("KP", "KP", "KP", "KP",
                  "KE",
                  "CC", "CC", "CC", "CC", "CC",
                  "Qu", "Qu", "Qu", "Qu", "Qu", "Qu", "Qu", "Qu",
                  "HW","HW", "HW")

    # space = ([
    #             [23, 14,  8, 42], 
    #             [31, 24, 10, 13],
    #             [50, 27,  5, 23],
    #             [13, 16, 50,  6],
    #             [14,  9,  3, 30],
    #             [47, 15, 21, 38],
    #             [22, 18, 47,  9],
    #             [55, 45, 35, 39],
    #             [34, 38, 16, 32],
    #             [24, 15, 36, 14],
    #             [46, 31, 41, 22]
    #         ],
    #          list(range(4)),
    #          [128], [256, 240, 224], [256, 240, 224], [512, 496, 480, 464], [512, 496, 480, 464],
    #          [16], [16], [16], [16], [16, 12, 8], [16, 12, 8], [16, 12, 8], [16, 12, 8],
    #          [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2])
    
    space = ([
                [32, 26, 29, 44],
                [34, 12, 19, 22],
                [6, 52, 24, 41],
                [28, 53, 22, 49],
                [16, 16, 41, 54],
                [18, 40, 15, 25],
                [48, 0, 13, 36],
                [13, 44, 4, 15],
                [48, 27, 45, 18],
                [12, 29, 23, 19],
                [34, 8, 41, 19],
                [0, 35, 16, 7],
                [12, 3, 50, 15],
                [48, 49, 10, 30],
                [41, 23, 18, 50],
                [49, 55, 15, 50],
                [7, 4, 28, 30],
                [14, 0, 34, 51],
                [50, 23, 28, 54],
                [40, 47, 50, 16],
                [1, 18, 19, 21],
                [51, 28, 28, 42],
                [37, 43, 14, 11],
                [46, 15, 49, 18],
                [2, 4, 0, 28],
                [33, 27, 24, 33],
                [2, 29, 17, 28],
                [17, 31, 30, 15],
                [5, 18, 50, 15],
                [37, 3, 14, 42],
                [13, 43, 38, 30],
                [36, 25, 15, 26],
                [39, 55, 51, 10],
                [55, 12, 5, 22],
                [5, 4, 18, 14],
                [26, 43, 1, 3],
                [1, 7, 9, 54],
                [16, 32, 39, 39],
                [9, 9, 49, 36],
                [47, 36, 53, 21],
                [23, 48, 17, 3],
                [46, 8, 3, 51],
                [47, 29, 22, 27],
                [53, 30, 21, 13],
                [1, 32, 49, 23],
                [34, 23, 53, 27],
                [53, 46, 54, 55],
                [37, 2, 0, 20],
                [47, 14, 30, 13],
                [6, 16, 29, 22],
                [20, 20, 1, 37],
                [15, 51, 49, 11],
                [28, 18, 36, 0],
                [35, 23, 1, 47],
                [16, 40, 3, 12],
                [40, 41, 8, 9],
                [15, 21, 10, 45],
                [20, 19, 47, 1],
                [34, 4, 50, 21],
                [52, 25, 49, 37],
                [16, 9, 10, 31],
                [51, 50, 11, 27],
                [53, 31, 5, 2],
                [25, 5, 1, 41],
                [5, 15, 18, 42],
                [38, 31, 3, 47],
                [51, 31, 47, 20],
                [1, 9, 30, 0],
                [46, 15, 38, 6],
                [16, 19, 5, 5],
                [7, 4, 29, 19],
                [41, 33, 47, 6],
                [21, 9, 1, 15],
                [21, 19, 37, 35],
                [10, 32, 42, 35],
                [53, 15, 55, 44],
                [15, 38, 22, 33],
                [13, 13, 18, 53],
                [23, 20, 27, 30],
                [43, 7, 49, 54],
                [0, 42, 3, 51],
                [32, 12, 44, 50],
                [31, 34, 0, 18],
                [19, 22, 47, 5],
                [39, 7, 33, 32],
                [17, 48, 39, 40],
                [12, 55, 28, 2],
                [3, 37, 28, 10],
                [16, 6, 38, 55],
                [46, 52, 13, 4],
                [8, 1, 49, 2],
                [50, 55, 39, 40],
                [48, 23, 13, 52],
                [36, 25, 4, 43],
                [39, 34, 27, 19],
                [34, 17, 19, 40],
                [48, 44, 8, 14],
                [6, 55, 6, 47],
                [15, 24, 1, 43],
                [29, 2, 26, 38],
            ],
             [3],
             [128], [256], [240], [512], [512],
             [16], [16], [16], [16], [8], [8], [8], [8],
             [-1], [0], [0])

    return space_name,space


def dna_analysis(dna, logger):
    pat_point, exp_point, ch_point, quant_point, comm_point = dna[0:4], dna[4], dna[5:10], dna[10:18], dna[18:21]

    pattern_33_space = pattern_sets_generate_3((3, 3), 3)

    for p in pat_point:
        logger.info("--------->Pattern 3-3 {}: {}".format(p, pattern_33_space[p].flatten()))
    logger.info("--------->Kernel Expand: {}".format(exp_point))
    logger.info("--------->Channel Cut: {}".format(ch_point))
    logger.info("--------->Qunatization: {}".format(quant_point))
    logger.info("--------->HW: {}".format(comm_point))

if __name__ == "__main__":
    # parser = argparse.ArgumentParser('Parser User Input Arguments')
    # parser.add_argument('--device', default='cpu', help='device')
    # args = parser.parse_args()

    args = train.parse_args()
    data_loader,data_loader_test = train.get_data_loader(args)


    model_name = "resnet18"


    hw_str = "70, 36, 64, 64, 7, 18, 6, 6"
    [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p] = [int(x.strip()) for x in hw_str.split(",")]
    HW = [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p]
    start_time = time.time()
    count = 60
    record = {}
    for i in range(count):
        model = globals()["resnet18"]()

        _, space = get_space()
        dna = []
        for selection in space:
            dna.append(random.choice(selection))
        print(dna)

        pat_point, exp_point, ch_point, quant_point, comm_point = dna[0:4], dna[4], dna[5:10], dna[10:18], dna[18:21]
        model = resnet_18_space(model, pat_point, exp_point, ch_point, quant_point, args)

        model = model.to(args.device)
        print("=" * 10, model_name, "performance analysis:")
        if W_p + comm_point[0] + I_p + comm_point[1] + O_p + comm_point[2] <= int(
                HW_constraints["r_Ports_BW"] / HW_constraints["BITWIDTH"]):
            total_lat = bottleneck_conv_only.get_performance(model, Tm, Tn, Tr, Tc, Tk, W_p + comm_point[0],
                                                             I_p + comm_point[1], O_p + comm_point[2], args.device)

            # latency.append(total_lat)
            print(total_lat)
        else:
            total_lat = -1
            print("-1")

        acc1,acc5,_ = train.main(args, dna, HW, data_loader, data_loader_test)
        record[i] = (acc5,total_lat)
        print("Random {}: acc-{}, lat-{}".format(i, acc5,total_lat))
        print(dna)
        print("=" * 100)

    print("="*100)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print("Exploration End, using time {}".format(total_time_str))
    for k,v in record.items():
        print(k,v)
    # print(min(latency), max(latency), sum(latency) / len(latency))

