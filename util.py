from __future__ import division

import torch
import torch.nn  as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = False):
    """
    for solve 2 problems
    p1. the processing such as thresholding by a object confidence, adding grid offsets to centers, applying anchors etc.
    p2. the dimensions of the prediction maps will be different
    convert detection feature map to 2-D tensor
    (19,19,B,(5+C)) --> (19X19XB, (5+C))
    :param prediction: the original output of module
    :param inp_dim:
    :param achors:
    :param num_classes:
    :param CUDA:
    :return:
    """
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs * anchors, grid_size * grid_size)  # (m, 85*3, 19*19) box1,box2 arranged in Horizontal
    prediction = prediction.transpose(1,2).contiguous()  # (m, 19*19, 85*3) contiguous means copy
    prediction = prediction.view(batch_size, grid_size * grid_size*num_anchors, bbox_attrs)  # (m, 19*19*3, 85)

    # add the grid offsets to the prediction od center coordination
    # add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).reshape(1, num_anchors).view(-1, 2).unsqueeze(0)  # ????

    prediction[:,:,:2] += x_y_offset

    # apply the anchors to the dimensions of the bounding




