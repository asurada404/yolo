from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from liu.yolo.util import predict_transform
#from .util import predict_transform

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
    2. shortcut   addition
    3. yolo
    4. route    concatenate
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
            conv = nn.Conv2d(pre_filters, filters, kernel_size, stride, pad, dilation= 1, bias = bias)
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
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
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
                # concatenate maps
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
# filename = "./cfg/yolov3.cfg"
# blocks = parse_cfd(filename)
# # print(blocks)
# print(create_modules(blocks))

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfd(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        print("forward begin....")
        modules = self.blocks[1:]
        outputs = {}  # cache the outputs for route layer
        # flag denote we encountered the first detection or not
        write = 0 #  the collector hasn't been initialized
        for i, module in enumerate(modules):

            module_type = (module["type"])
            #print("i: ", i, "module name: ", module_type, "x shape,", x.shape)

            # convolutional and upsample
            if module_type == "convolutional" or module_type == "upsample":
                #print(x.shape)
                #print(self.module_list[i])
                x = self.module_list[i](x)

            # route layer / shortcut layer
            # route layer
            # concatenate two feature maps along the depth (channel)
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
                if (layers[0]) > 0:     #  why??
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]  # int + tuple???
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)

            # shortcut layer
            # add the previous layer
            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i + from_]

            # yolo layer
            # map (19,19,5,85)
            # e.g  map[5,6, (5+C): 2*(5+c)]: access the second bounding of cell at (5,6)
            elif module_type == "yolo":
                anchors = self.module_list[i][0].anchors
                # the input size
                inp_dim = int(self.net_info["height"])
                # the number of the classes
                num_classes = int(module["classes"])

                # transform
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x),1)
            outputs[i] = x
        return detections

    def load_weights(self, weightfile):
        fp = open(weightfile, "rb")
        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        # load header
        header = np.fromfile(fp, dtype=np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        # load weight
        weights = np.fromfile(fp, dtype = np.float32)
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            if module_type == "convolutional":
                print("convolutional, ", i)
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]
                if batch_normalize:  # wait!!!!!
                    bn = model[1]
                    # get the number  of the weights of the BN layer
                    num_bn_biases = bn.bias.numel()
                    #load weight
                    bn_biases = torch.from_numpy(weights[ptr : ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr : ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr : ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var =  torch.from_numpy(weights[ptr : ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # cast the loaded weight to model
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_biases.view_as(bn.running_mean)
                    bn_running_var = bn_weights.view_as(bn.running_var)

                    # copy data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else: # no batch normal
                    num_biases = conv.bias.numel()
                    # load weights
                    conv_biases = torch.from_numpy(weights[ptr:ptr + num_biases])
                    ptr = ptr  + num_biases

                    #reshape the weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # copy data
                    conv.bias.data.copy_(conv_biases)

                #load convolutional layer's weight
                num_weights = conv.weight.numel()

                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

# # test load weights
# weight_path = "./weights/yolov3.weights"
# cfg_path = "./cfg/yolov3.cfg"
# model = Darknet(cfg_path)
# model.load_weight(weight_path)

# test darknet model
def get_test_input(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img, (416, 416))
    img_ = img[:,:,::-1].transpose((2,0,1)) # BGR -> RGB  and  HWC -> CHW
    img_ = img_[np.newaxis, :,:,:]/255.0 # add chanel as batch and normalize
    img_ = torch.from_numpy(img_).float()
    print(img_.shape)
    img_ = Variable(img_)
    return img_

# # test forward
# cfg_filename = "./cfg/yolov3.cfg"
# model = Darknet(cfg_filename)
# img_filename = "./images/dog-cycle-car.png"
# inp = get_test_input(img_filename)
# pred = model(inp, torch.cuda.is_available())
# print(pred)










