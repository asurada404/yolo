from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def parse_cfd(cfgfile):
    """
    parse cfg file to save in dict
    :param cfgfile:
    :return:
    """
    file = open(cfgfile, 'r')
    lines = file.read().split("\n")
    lines = [x for x in lines if len(x) > 0]  # remove the empty line
    lines = [x for x in lines if x[0] != "#"]  # remove the comment lines
    lines = [x.rstrip().lstrip() for x in lines]  # remove the fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # new block
            if len(block) > 0:  # add previous block
                blocks.append(block)
                block = {}  # re-init the block
            block["type"] = line[1:-1].rstrip()  # add type of the block
        else:
            key, val = line.split("=")
            block[key.rstrip()] = val.lstrip()
    blocks.append(block)
    return blocks
    # print(blocks)


# # test parse_cfd
# filename = "./cfg/yolov3.cfg"
# parse_cfd(filename)

def create_modules(blocks):
    """
    construct model
    there are 5 types of layers
    1. convolutional
    2. shortcut
    3. yolo
    4. route
    5. unsample
    another net
    :param blocks:
    :return:
    """
    net_info = blocks[0]
    module_list = nn.ModuleList()
    pre_filters = 3  # the original channel of the images
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        # print(len(blocks[1:]))
        module = nn.Sequential()

        # convolutional
        if x["type"] == "convolutional":
            # print("convolutional")
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2  # why?????
            else:
                pad = 0

            # add the convolutional layer
            conv = nn.Conv2d(pre_filters, filters, kernel_size, stride, pad, bias)
            module.add_module("conv_{0}".format(index), conv)

            # add Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # check the activation  Linear or Leaky
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

        # unsample layer
        elif x["type"] == "upsample":
            # print("upsample")
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            module.add_module("upsample_{0}".format(index), upsample)

        # # route layer     ## don't know !!!!!!
        elif x["type"] == "route":
            # print("route")
            x["layers"] = x["layers"].split(",")
            # start of the route
            start = int(x["layers"][0])
            # end , if end exists
            try:
                end = int(x["layers"][1])
            except:
                end = 0

            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]



        # shortcut
        elif x["type"] == "shortcut":
            # print("shortcut")
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        # yolo
        elif x["type"] == "yolo":
            # print("yolo")
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{0}".format(index), detection)

        module_list.append(module)
        pre_filters = filters
        output_filters.append(filters)

    return net_info, module_list


# test create module
filename = "./cfg/yolov3.cfg"
blocks = parse_cfd(filename)
# print(blocks)
print(create_modules(blocks))
