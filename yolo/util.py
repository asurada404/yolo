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
    #print("prediction:" ,prediction.shape)
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)  # original image -> feature map ; e.g: 416x416 -> 13x13 stride = 32

    grid_size = inp_dim // stride
    print("new", inp_dim, stride, grid_size)
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    #print(batch_size, bbox_attrs * anchors, )
    #print(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)  # (m, 85*3, 19*19) box1,box2 arranged in Horizontal
    prediction = prediction.transpose(1,2).contiguous()  # (m, 19*19, 85*3) contiguous means copy
    prediction = prediction.view(batch_size, grid_size * grid_size*num_anchors, bbox_attrs)  # (m, 19*19*3, 85)

    x_y_pre  = prediction[:,:,:2]  # the center of the anchor
    w_h_pre = prediction[:,:,2:4]  # the width and height of the anchor
    class_pre = prediction[:,:,5 : 5 + num_classes]  # the one-hot result of class
    # add the grid offsets to the prediction od center coordination
    # add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    # the process of the transform can be seen in predict_transform_process.py
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    # apply the anchors to the dimensions of the bounding
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    # sigmoid activation to the class scores
    prediction[:,:,5:5 + num_classes] = torch.sigmoid((prediction[:,:,5:5 + num_classes]))

    # resize the detections map to the size of the input image
    prediction[:,:,:4] *= stride
    return prediction


def unique(tensor):
    """
    multiple true detections of the same class
    :param tensor:
    :return:
    """
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def bbox_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b1_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min = 0) * torch.clamp( inter_rect_y2  - inter_rect_y1, min = 0)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y1 - b1_y2 + 1)  #  + 1 why???
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y1 - b2_y2 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area)
    return iou


def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    # prob_score
    prob_score = prediction[:,:,4]
    print("prediction shape : ", prediction.shape)
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    print("conf_mask: ", conf_mask.shape)
    prediction = prediction * conf_mask
    print("prediction shape : ", prediction.shape)

    # Non-maximum Suppression
    # transform x,y,w,h to x1,y1,x2,y2 (bounding box)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:,:,:4] = box_corner[:,:,:4]

    # loop over the batch size
    batch_size = prediction.size(0)
    write = False  # ????
    for ind in range(batch_size):
        image_pred  = prediction[ind]
        max_conf, max_conf_score = torch.max(image_pred[:, 5: 5+ num_classes], 1)
        print("max_conf shape: ", max_conf.shape)
        print("max_conf_score shape: ", max_conf_score.shape)
        max_conf = max_conf.float().unsqueeze(1)
        print("max_conf shape: ", max_conf.shape)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        # gid rid of bounding box with low confidence
        non_zero_ind = (torch.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1,7)
        except:
            continue

        if image_pred.shape(0) == 0:
            continue

        img_classes = unique(image_pred[:,-1]) # class index

        for cls in img_classes:
            # perform NMS for each class
            # extract the detections of a particular class
            cls_mask  = image_pred_ * (image_pred_[:,-1] == cls).float().unsqueeze()
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)

            # sort the detections
            conf_sort_index = torch.sort(image_pred_class[:,-4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)

            for i in range(idx):
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
                except IndexError:
                    break

                iou_mask = (ious < nms_conf).float().unsequeeze(1)
                image_pred_class[i+1:] *= iou_mask

                # remove the non-zero entries
                non_zero_ind  = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)
                batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)

                seq = batch_ind, image_pred_class
                if not write:
                    output = torch.cat(seq, 1)
                    write = True
                else:
                    out = torch.cat(seq, 1)
                    output = torch.cat((output, out))

        try :
            return output
        except:
            return 0






























