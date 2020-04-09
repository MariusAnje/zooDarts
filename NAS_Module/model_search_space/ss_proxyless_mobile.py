import sys
from torchvision import models

from torchvision.models import *
sys.path.append("../")
sys.path.append("../../Interface")
sys.path.append("../../Performance_Model")
from model_modify import *
import train
import random

# [1,22,49,54], 3, [100,210,210,470,470]
def proxyless_mobile_space(model, dna, args):

    global p3size
    global p5size

    pattern_3_3_idx = dna[0:4]
    pattern_5_5_idx = dna[4:8]
    pattern_do_or_not = dna[8:19]
    q_list = dna[19:]

    # pattern_idx = [0, 1, 2, 3]

    pattern_55_space = pattern_sets_generate_3((5, 5),p5size)
    pattern_55 = {}
    i = 0
    for idx in pattern_5_5_idx:
        pattern_55[i] = pattern_55_space[idx].reshape((5, 5))
        i+=1
    layer_names_55 = [
        "blocks.1.mobile_inverted_conv.depth_conv.conv",
        "blocks.7.mobile_inverted_conv.depth_conv.conv",
        "blocks.8.mobile_inverted_conv.depth_conv.conv",
        "blocks.10.mobile_inverted_conv.depth_conv.conv",
        "blocks.11.mobile_inverted_conv.depth_conv.conv",
        "blocks.12.mobile_inverted_conv.depth_conv.conv",
        "blocks.14.mobile_inverted_conv.depth_conv.conv",
        "blocks.15.mobile_inverted_conv.depth_conv.conv",
        "blocks.16.mobile_inverted_conv.depth_conv.conv"
    ]

    # print(model)
    #
    # for n,p in model.named_parameters():
    #     print(n)
    # model.state_dict()['blocks.11.mobile_inverted_conv.depth_conv.conv.weight']
    # sys.exit(0)

    layer_names_55_select = []
    for i in range(9):
        if pattern_do_or_not[i] == 1:
            layer_names_55_select.append(layer_names_55[i])


    pattern_33_space = pattern_sets_generate_3((3, 3), p3size)
    pattern_33 = {}
    i = 0
    for idx in pattern_3_3_idx:
        pattern_33[i] = pattern_33_space[idx].reshape((3, 3))
        i += 1

    layer_33_names = ["first_conv.conv","blocks.0.mobile_inverted_conv.depth_conv.conv"]

    layer_names_33_select = []
    for i in range(2):
        if pattern_do_or_not[i+9] == 1:
            layer_names_33_select.append(layer_33_names[i])

    Kernel_Patter(model, layer_names_55_select, pattern_55, args)
    Kernel_Patter(model, layer_names_33_select, pattern_33, args)

    # Change all layer to 16 bit
    quan_paras = {}

    quan_paras["first_conv.conv"] = [1, 15, True]
    quan_paras["blocks.0.mobile_inverted_conv.depth_conv.conv"] = [3, 13, True]
    quan_paras["blocks.0.mobile_inverted_conv.point_linear.conv"] = [2, 14, True]
    quan_paras["blocks.1.mobile_inverted_conv.inverted_bottleneck.conv"] = [2, 14, True]
    quan_paras["blocks.1.mobile_inverted_conv.depth_conv.conv"] = [1, 15, True]
    quan_paras["blocks.1.mobile_inverted_conv.point_linear.conv"] = [2, 14, True]
    quan_paras["blocks.2.mobile_inverted_conv.inverted_bottleneck.conv"] = [1, 15, True]
    quan_paras["blocks.2.mobile_inverted_conv.depth_conv.conv"] = [1, 15, True]
    quan_paras["blocks.2.mobile_inverted_conv.point_linear.conv"] = [1, 15, True]
    quan_paras["blocks.5.mobile_inverted_conv.inverted_bottleneck.conv"] = [2, 14, True]
    quan_paras["blocks.5.mobile_inverted_conv.depth_conv.conv"] = [1, 15, True]
    quan_paras["blocks.5.mobile_inverted_conv.point_linear.conv"] = [2, 14, True]
    quan_paras["blocks.6.mobile_inverted_conv.inverted_bottleneck.conv"] = [1, 15, True]
    quan_paras["blocks.6.mobile_inverted_conv.depth_conv.conv"] = [1, 15, True]
    quan_paras["blocks.6.mobile_inverted_conv.point_linear.conv"] = [1, 15, True]
    quan_paras["blocks.7.mobile_inverted_conv.inverted_bottleneck.conv"] = [1, 15, True]
    quan_paras["blocks.7.mobile_inverted_conv.depth_conv.conv"] = [1, 15, True]
    quan_paras["blocks.7.mobile_inverted_conv.point_linear.conv"] = [1, 15, True]
    quan_paras["blocks.8.mobile_inverted_conv.inverted_bottleneck.conv"] = [1, 15, True]
    quan_paras["blocks.8.mobile_inverted_conv.depth_conv.conv"] = [1, 15, True]
    quan_paras["blocks.8.mobile_inverted_conv.point_linear.conv"] = [1, 15, True]
    quan_paras["blocks.9.mobile_inverted_conv.inverted_bottleneck.conv"] = [1, 15, True]
    quan_paras["blocks.9.mobile_inverted_conv.depth_conv.conv"] = [1, 15, True]
    quan_paras["blocks.9.mobile_inverted_conv.point_linear.conv"] = [1, 15, True]
    quan_paras["blocks.10.mobile_inverted_conv.inverted_bottleneck.conv"] = [1, 15, True]
    quan_paras["blocks.10.mobile_inverted_conv.depth_conv.conv"] = [1, 15, True]
    quan_paras["blocks.10.mobile_inverted_conv.point_linear.conv"] = [1, 15, True]
    quan_paras["blocks.11.mobile_inverted_conv.inverted_bottleneck.conv"] = [1, 15, True]
    quan_paras["blocks.11.mobile_inverted_conv.depth_conv.conv"] = [1, 15, True]
    quan_paras["blocks.11.mobile_inverted_conv.point_linear.conv"] = [1, 15, True]
    quan_paras["blocks.12.mobile_inverted_conv.inverted_bottleneck.conv"] = [1, 15, True]
    quan_paras["blocks.12.mobile_inverted_conv.depth_conv.conv"] = [1, 15, True]
    quan_paras["blocks.12.mobile_inverted_conv.point_linear.conv"] = [1, 15, True]
    quan_paras["blocks.13.mobile_inverted_conv.inverted_bottleneck.conv"] = [1, 15, True]
    quan_paras["blocks.13.mobile_inverted_conv.depth_conv.conv"] = [1, 15, True]
    quan_paras["blocks.13.mobile_inverted_conv.point_linear.conv"] = [1, 15, True]
    quan_paras["blocks.14.mobile_inverted_conv.inverted_bottleneck.conv"] = [1, 15, True]
    quan_paras["blocks.14.mobile_inverted_conv.depth_conv.conv"] = [1, 15, True]
    quan_paras["blocks.14.mobile_inverted_conv.point_linear.conv"] = [1, 15, True]
    quan_paras["blocks.15.mobile_inverted_conv.inverted_bottleneck.conv"] = [1, 15, True]
    quan_paras["blocks.15.mobile_inverted_conv.depth_conv.conv"] = [1, 15, True]
    quan_paras["blocks.15.mobile_inverted_conv.point_linear.conv"] = [1, 15, True]
    quan_paras["blocks.16.mobile_inverted_conv.inverted_bottleneck.conv"] = [1, 15, True]
    quan_paras["blocks.16.mobile_inverted_conv.depth_conv.conv"] = [1, 15, True]
    quan_paras["blocks.16.mobile_inverted_conv.point_linear.conv"] = [1, 15, True]
    quan_paras["blocks.17.mobile_inverted_conv.inverted_bottleneck.conv"] = [1, 15, True]
    quan_paras["blocks.17.mobile_inverted_conv.depth_conv.conv"] = [1, 15, True]
    quan_paras["blocks.17.mobile_inverted_conv.point_linear.conv"] = [1, 15, True]
    quan_paras["blocks.18.mobile_inverted_conv.inverted_bottleneck.conv"] = [1, 15, True]
    quan_paras["blocks.18.mobile_inverted_conv.depth_conv.conv"] = [1, 15, True]
    quan_paras["blocks.18.mobile_inverted_conv.point_linear.conv"] = [1, 15, True]
    quan_paras["blocks.19.mobile_inverted_conv.inverted_bottleneck.conv"] = [1, 15, True]
    quan_paras["blocks.19.mobile_inverted_conv.depth_conv.conv"] = [1, 15, True]
    quan_paras["blocks.19.mobile_inverted_conv.point_linear.conv"] = [1, 15, True]
    quan_paras["blocks.20.mobile_inverted_conv.inverted_bottleneck.conv"] = [1, 15, True]
    quan_paras["blocks.20.mobile_inverted_conv.depth_conv.conv"] = [1, 15, True]
    quan_paras["blocks.20.mobile_inverted_conv.point_linear.conv"] = [1, 15, True]
    quan_paras["blocks.21.mobile_inverted_conv.inverted_bottleneck.conv"] = [1, 15, True]
    quan_paras["blocks.21.mobile_inverted_conv.depth_conv.conv"] = [1, 15, True]
    quan_paras["blocks.21.mobile_inverted_conv.point_linear.conv"] = [1, 15, True]
    quan_paras["feature_mix_layer.conv"] = [1, 15, True]


    Kenel_Quantization(model, quan_paras.keys(), quan_paras)



    # Modify layers that is dominated by loading weight
    quan_paras = {}
    quan_paras["blocks.17.mobile_inverted_conv.depth_conv.conv"] = [1, q_list[0], True]
    quan_paras["blocks.17.mobile_inverted_conv.point_linear.conv"] = [1, q_list[1], True]
    quan_paras["blocks.18.mobile_inverted_conv.inverted_bottleneck.conv"] = [1, q_list[2], True]
    quan_paras["blocks.18.mobile_inverted_conv.point_linear.conv"] = [1, q_list[3], True]

    quan_paras["blocks.19.mobile_inverted_conv.inverted_bottleneck.conv"] = [1, q_list[4], True]
    quan_paras["blocks.19.mobile_inverted_conv.point_linear.conv"] = [1, q_list[5], True]
    quan_paras["blocks.20.mobile_inverted_conv.inverted_bottleneck.conv"] = [1, q_list[6], True]
    quan_paras["blocks.20.mobile_inverted_conv.point_linear.conv"] = [1, q_list[7], True]

    quan_paras["blocks.21.mobile_inverted_conv.inverted_bottleneck.conv"] = [1, q_list[8], True]
    quan_paras["blocks.21.mobile_inverted_conv.point_linear.conv"] = [1, q_list[9], True]
    quan_paras["feature_mix_layer.conv"] = [1, q_list[10], True]

    Kenel_Quantization(model, quan_paras.keys(), quan_paras)



    # layer_name = "layers.12.3.layers.3"
    # seq = layer_name.split(".")
    # (pre_attr, last_attr, last_not_digit) = get_last_attr_idx(model, seq)
    # print(last_attr[3].check_layer())

    # print(model)
    return model

def get_space():

    global p3size
    global p5size

    p3size = 3
    p5size = 6

    space_name = ("KP-3","KP-3","KP-3","KP-3",
                  "KP-5","KP-5","KP-5","KP-5",
                  "KP5 S or N", "KP5 S or N", "KP5 S or N", "KP5 S or N", "KP5 S or N",
                  "KP5 S or N", "KP5 S or N", "KP5 S or N", "KP5 S or N",
                  "KP3 S or N", "KP3 S or N",
                  "Quan","Quan","Quan","Quan",
                  "Quan","Quan","Quan","Quan",
                  "Quan","Quan","Quan")

    pattern_33_space = pattern_sets_generate_3((3, 3), p3size)
    pattern_55_space = pattern_sets_generate_3((5, 5), p5size)

    p3num = len(pattern_33_space.keys())
    p5num =  len(pattern_55_space.keys())

    space = (list(range(p3num)),list(range(p3num)),list(range(p3num)),list(range(p3num)),
             list(range(p5num)), list(range(p5num)), list(range(p5num)), list(range(p5num)),
             [0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],
             list(range(4, 15, 4)), list(range(4, 15, 4)), list(range(4, 15, 4)), list(range(4, 15, 4)),
             list(range(4, 15, 4)), list(range(4, 15, 4)), list(range(4, 15, 4)), list(range(4, 15, 4)),
             list(range(4, 15, 4)), list(range(4, 15, 4)), list(range(4, 15, 4)))
    return space_name,space

def dna_analysis(dna,logger):
    global p3size
    global p5size

    pattern_3_3_idx = dna[0:4]
    pattern_5_5_idx = dna[4:8]
    pattern_do_or_not = dna[8:19]
    q_list = dna[19:]

    pattern_33_space = pattern_sets_generate_3((3, 3), p3size)
    pattern_55_space = pattern_sets_generate_3((5, 5), p5size)


    for p in pattern_3_3_idx:
        logger.info("--------->Pattern 3-3 {}: {}".format(p, pattern_33_space[p].flatten()))
    for p in pattern_5_5_idx:
        logger.info("--------->Pattern 5-5 {}: {}".format(p, pattern_55_space[p].flatten()))
    logger.info("--------->Weight Pruning or Not: {}".format(pattern_do_or_not))
    logger.info("--------->Quantization Selection: {}".format(q_list))




if __name__ == "__main__":
    # parser = argparse.ArgumentParser('Parser User Input Arguments')
    # parser.add_argument(
    #     '-c', '--cconv',
    #     default="100, 16, 32, 32, 3, 10, 10, 10",
    #     help="hardware desgin of cconv",
    # )
    # parser.add_argument(
    #     '-dc', '--dconv',
    #     default="832, 1, 32, 32, 7, 10, 10, 10",
    #     help="hardware desgin of cconv",
    # )
    #
    # parser.add_argument('--device', default='cpu', help='device')
    # args = parser.parse_args()

    args = train.parse_args()
    data_loader, data_loader_test = train.get_data_loader(args)

    model_name = "proxyless_mobile"
    model = torch.hub.load('mit-han-lab/ProxylessNAS', model_name, pretrained=True)
    HW1 = [int(x.strip()) for x in args.dconv.split(",")]
    HW2 = [int(x.strip()) for x in args.cconv.split(",")]


    count = 10

    latency = []

    for i in range(count):

        _, space = get_space()
        dna = []
        for selection in space:
            dna.append(random.choice(selection))
        print(dna)

        pattern_3_3_idx = dna[0:4]
        pattern_5_5_idx = dna[4:8]
        pattern_do_or_not = dna[8:19]
        q_list = dna[19:]

        model = proxyless_mobile_space(model, dna, args)
        model = model.to(args.device)
        print("=" * 10, model_name, "performance analysis:")
        total_lat = bottlenect_conv_dconv.get_performance(model, HW1, HW2, args.device)
        print(total_lat)
        latency.append(total_lat)

        acc1, acc5, _ = train.main(args, dna, HW, data_loader, data_loader_test)
        print(acc1,acc5,total_lat)

    print(min(latency),max(latency),sum(latency)/len(latency))

