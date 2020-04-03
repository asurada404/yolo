from __future__ import division

import torch
import torch.nn  as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


def unique(tensor):
    """
    multiple true detections of the same class
    :param tensor:  class_confidence_max_index (10647)
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
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min = 0) * torch.clamp( inter_rect_y2  - inter_rect_y1 + 1, min = 0)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)  #  + 1 why???
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area)
    return iou


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = False):
    """
    for solve 2 problems
    p1. the processing such as thresholding by a object confidence, adding grid offsets to centers, applying anchors etc.
    p2. the dimensions of the prediction maps will be different
    convert detection feature map to 2-D tensor
    (13,13,B,(5+C)) --> (13X13XB, (5+C))
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
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
    #print(batch_size, bbox_attrs * anchors, )
    #print(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)  # (m, 85*3, 13*13) box1,box2 arranged in Horizontal
    prediction = prediction.transpose(1,2).contiguous()  # (m, 13*13, 85*3) contiguous means copy
    prediction = prediction.view(batch_size, grid_size * grid_size*num_anchors, bbox_attrs)  # (m, 13*13*3, 85)

    x_y_pre  = prediction[:,:,:2]  # the center of the anchor
    w_h_pre = prediction[:,:,2:4]  # the width and height of the anchor
    class_pre = prediction[:,:,5 : 5 + num_classes]  # the one-hot result of class

    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])


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
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0) #[1,13*13*3,2]

    prediction[:,:,:2] += x_y_offset

    # apply the anchors to the dimensions of the bounding
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)  # [1,13*13*3,2]
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors  # [1,13*13*3,2]

    # sigmoid activation to the class scores
    prediction[:,:,5:5 + num_classes] = torch.sigmoid((prediction[:,:,5:5 + num_classes]))

    # resize the detections map to the size of the input image
    prediction[:,:,:4] *= stride
    return prediction


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)

    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]
        print("image_pred", image_pred.shape)
        # image Tensor
        # #confidence threshholding
        # #NMS

        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)


        non_zero_ind = (torch.nonzero(image_pred[:, 4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except:
            continue

        if image_pred_.shape[0] == 0:
            continue
        print(".....image_pred_>>>>>>>", image_pred_.shape)
        print("image_pred_", image_pred_.shape)
        # Get the various classes detected in the image
        img_classes = unique(image_pred_[:, -1])  # -1 index holds the class index

        print("img_classes shape", img_classes)
        print("image_pred_", image_pred_.shape)
        for cls in img_classes:
            print("cls", cls)
            # perform NMS

            # get the detections with one particular class
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)  # Number of detections

            for i in range(idx):
                # Get the IOUs of all boxes that come after the one we are looking at
                # in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                except ValueError:
                    break

                except IndexError:
                    break

                # Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i + 1:] *= iou_mask

                # Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(
                ind)  # Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    try:
        return output
    except:
        return 0


# def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
#     """
#
#     :param prediction:   (m, 10647, 85)
#     :param confidence:
#     :param num_classes:
#     :param nms_conf:
#     :return:
#     """
#     # prob_score
#     prob_score = prediction[:,:,4]
#     print("prediction shape : ", prediction.shape)
#     conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)  #  (m, 10647, 1)
#     print("conf_mask: ", conf_mask.shape)
#     prediction = prediction * conf_mask  #  (m, 10647, 85) * (m, 10647, 1)
#     print("prediction shape : ", prediction.shape)  # (m, 10647, 85)
#
#     # Non-maximum Suppression
#     # transform x,y,w,h to x1,y1,x2,y2 (bounding box)
#     box_corner = prediction.new(prediction.shape)
#     box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
#     box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
#     box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
#     box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
#     prediction[:,:,:4] = box_corner[:,:,:4]
#
#     # loop over the batch size
#     batch_size = prediction.size(0)
#     write = False  # ????
#     for ind in range(batch_size):
#
#         image_pred  = prediction[ind]
#         print("image_pred", image_pred.shape)
#         max_conf, max_conf_score = torch.max(image_pred[:, 5: 5+ num_classes], 1)
#         max_conf = max_conf.float().unsqueeze(1)
#         max_conf_score = max_conf_score.float().unsqueeze(1)
#         seq = (image_pred[:, :5], max_conf, max_conf_score)
#         image_pred = torch.cat(seq, 1) # (10647, 7) [x1,y1,x2,y2, object_confidence, class_confidence_max, class_confidence_max_index]
#
#         # gid rid of bounding box with low confidence
#         non_zero_ind = (torch.nonzero(image_pred[:,4]))
#         try:
#             image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1,7)  # (?, 7)
#         except:
#             continue
#         if image_pred_.shape[0]== 0:
#             continue
#         print("image_pred_", image_pred_.shape)
#         img_classes = unique(image_pred_[:,-1]) # class index
#         print("img_classes shape", img_classes)
#         print("image_pred_", image_pred_.shape)
#         for cls in img_classes:
#             print("cls", cls)
#             # perform NMS for each class
#             # extract the detections of a particular class
#             cls_mask = image_pred_ * (image_pred_[:,-1] == cls).float().unsqueeze(1) #(?, 7)
#             class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()  #
#             image_pred_class = image_pred_[class_mask_ind].view(-1,7)
#
#             # sort the detections
#             conf_sort_index = torch.sort(image_pred_class[:,4], descending=True)[1]
#             image_pred_class = image_pred_class[conf_sort_index]
#             idx = image_pred_class.size(0)
#
#             for i in range(idx):
#                 try:
#                     ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
#                 except ValueError:
#                     break
#                 except IndexError:
#                     break
#
#                 iou_mask = (ious < nms_conf).float().unsqueeze(1)
#                 image_pred_class[i+1:] *= iou_mask
#
#                 # remove the non-zero entries
#                 non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
#                 image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)
#             batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
#
#             seq = batch_ind, image_pred_class
#             if not write:
#                 output = torch.cat(seq, 1)
#                 write = True
#             else:
#                 out = torch.cat(seq, 1)
#                 output = torch.cat((output, out))
#     try :
#         return output
#     except:
#         return 0


def letterbox_image(img, inp_dim):
    """
    resize the image keep aspect ratio
    :param img:
    :param inp_dim:
    :return:  the image which have the input size 416 416 3
    """
    img_w, img_h = img.shape[1], img.shape[0]   # img: HWC
    w,h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)  # WHC 416 416 3
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)  # HWC 416 416 3
    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image
    return canvas

def prep_image(img, inp_dim):
    """
    prepare image
    :param img:
    :param inp_dim:
    :return:
    """
    img = letterbox_image(img, (inp_dim, inp_dim))  # ( 416,416,3)
    img = img[:,:,::-1].transpose((2,0,1)).copy()    # (416,416,3) --> (3,416,416) HWC --> CHW (BGR --> RGB)
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)  # (3,416,416) --> (1,3,416,416) BCHW
    return img

def load_classes(filepath):
    fp = open(filepath)
    names = fp.read().split("\n")[:-1]
    return names


























