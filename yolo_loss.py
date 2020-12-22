import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from utils import *

def bbox_iou(box1, box2, gridSizeX, gridSizeY):
    box1[2] = 0.25
    box2[2] = 0.25
    mx = max(min(box1[0] - box1[2], box2[0] - box2[2]), 0)
    Mx = min(max(box1[0] + box1[2], box2[0] + box2[2]), gridSizeX)
    my = max(min(box1[1] - box1[2], box2[1] - box2[2]), 0)
    My = min(max(box1[1] + box1[2], box2[1] + box2[2]), gridSizeY)
    w1 = box1[2] * 2
    h1 = box1[2] * 2
    w2 = box2[2] * 2
    h2 = box2[2] * 2

    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    if cw <= 0 or ch <= 0:
        return 0.0
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return (carea / (uarea + 1e-12)).item()

def make_targets(pred_boxes, target, best_ns, s, num_classes, nH, nW, sil_thresh):
    nB = target.size(0)
    nC = num_classes

    mask = torch.zeros([nB, nH, nW], dtype=torch.uint8)
    tx = torch.zeros(nB, nH, nW)
    ty = torch.zeros(nB, nH, nW)
    tconf = torch.zeros(nB, nH, nW)
    nCorrect = 0
    for b in range(nB):
        for t in range(1):
            if target[b][t][0] == 0:
                break
            gx = target[b][t][0] * nW
            gy = target[b][t][1] * nH
            gi = int(gx)
            gj = int(gy)
            best_n = best_ns[b][s][t].item()

            gt_box = [gx, gy, 1]
            pred_box = pred_boxes[b, gj, gi]

            mask[b][gj][gi] = 1
            tx[b][gj][gi] = gx - gi
            ty[b][gj][gi] = gy - gj

            iou = bbox_iou(gt_box, pred_box.data, nW, nH)  # best_iou
            tconf[b][gj][gi] = iou

            if pred_box[2] > sil_thresh:
                nCorrect = nCorrect + 1

    return nCorrect, mask, tx, ty, tconf

def YoloLoss(outputs, target, num_classes, seen, device):
    out_channels = (6 + num_classes) * 3
    coord_scale = 10
    noobject_scale = 0.2
    object_scale = 0.8
    thresh = 0.5

    nB = outputs[0].size(0)
    nC = num_classes

    nGT = 0
    nCorrect = [0, 0, 0]
    nProposals = [0, 0, 0]
    best_n = torch.zeros([nB, len(outputs), 1], dtype=torch.uint8)
    for b in range(nB):
        for t in range(1):
            if target[b][t][0] == 0:
                break
            nGT += 1
    loss = []
    for s in range(len(outputs)):
        nH = outputs[s].size(2)
        nW = outputs[s].size(3)
        print(outputs[s].shape)
        output = outputs[s].view(nB, out_channels, nH, nW).permute(0, 2, 3, 1).contiguous()
          # Get outputs
        x = torch.sigmoid(output[..., 0])  # Center x
        y = torch.sigmoid(output[..., 1])  # Center y
        w = torch.sigmoid(output[..., 2])  #
        h = torch.sigmoid(output[..., 3])  #
        angle = torch.sigmoid(output[..., 4])  #
        conf = torch.sigmoid(output[..., -1])  # Conf

        pred_boxes = torch.FloatTensor(3, nB, nH, nW).to(device)
        grid_x = torch.linspace(0, nW - 1, nW).repeat(nH, 1).repeat(nB, 1, 1).view(nB, nH, nW).to(device)
        grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB, 1, 1).view(nB, nH, nW).to(device)
        # pred_boxes[2] = torch.exp(w.data) * anchor_w
        # pred_boxes[3] = torch.exp(h.data) * anchor_h
        pred_boxes[0] = x.data + grid_x
        pred_boxes[1] = y.data + grid_y
        pred_boxes[2] = conf.data
        pred_boxes = pred_boxes.permute(1, 2, 3, 0).contiguous().cpu()

        nCorrect[s], mask, tx, ty, tconf = \
            make_targets(pred_boxes, target.data, best_n, s, nC, nH, nW, thresh)

        nProposals[s] = torch.sum(conf > 0.5).item()
        # print(conf.shape, conf)

        tx = tx.to(device)
        ty = ty.to(device)
        tconf = tconf.to(device)
        mask = mask.to(device)
        if len(tconf[mask == 1]) != 0:
            loss_x = coord_scale * nn.MSELoss(reduction='sum')(torch.exp(x[mask == 1]), torch.exp(tx[mask == 1]))
            loss_y = coord_scale * nn.MSELoss(reduction='sum')(torch.exp(y[mask == 1]), torch.exp(ty[mask == 1]))
            loss_angle = coord_scale * nn.MSELoss(reduction='sum')(torch.exp(y[mask == 1]), torch.exp(ty[mask == 1]))
            loss_conf = object_scale * nn.MSELoss(reduction='sum')(conf[mask == 1], tconf[mask == 1]) + \
                        noobject_scale * nn.MSELoss(reduction='sum')(conf[mask == 0], tconf[mask == 0])


            loss_tot = loss_x + loss_y + loss_angle + loss_angle+loss_conf
            loss.append(loss_tot)
            print('%d: nGT: %d, recall: %d, proposals: %d, loss: x %f, y %f, angle %f, conf %f, total %f' % (
                seen, nGT, nCorrect[s], nProposals[s], loss_x.item(), loss_y.item(),  loss_angle.item(), loss_conf.item(), loss[s].item()))
        else:
            loss_tot = noobject_scale * nn.MSELoss(reduction='sum')(conf[mask == 0], tconf[mask == 0])
            loss.append(loss_tot)
            print('%d: nGT: %d, recall: %d, proposals: %d, loss: noobj %f' % (
                seen, nGT, nCorrect[s], nProposals[s], loss[s].item()))

    return loss, nGT, nCorrect, nProposals

def test():
    output0 = torch.randn(1, 3, 12, 12)
    output1 = torch.randn(1, 3, 24, 24)
    output2 = torch.randn(1, 3, 48, 48)
    target = torch.tensor([0.54,0.51]).float()
    target = target[None, None,:]
    Outputs = [output0, output1, output2]
    loss, nGT, nCorrect, nProposals = YoloLoss(Outputs, target, 1,
                                               1, 'cpu')

# test()