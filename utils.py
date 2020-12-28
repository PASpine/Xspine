import sys
import os
import time
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import torch.nn.functional as F

import itertools
import struct  # get_image_size
import imghdr  # get_image_size


def sigmoid(x):
    return 1.0 / (math.exp(-x) + 1.)


def softmax(x):
    x = torch.exp(x - torch.max(x))
    x = x / x.sum()
    return x


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0] - box1[2] / 2.0, box2[0] - box2[2] / 2.0)
        Mx = max(box1[0] + box1[2] / 2.0, box2[0] + box2[2] / 2.0)
        my = min(box1[1] - box1[3] / 2.0, box2[1] - box2[3] / 2.0)
        My = max(box1[1] + box1[3] / 2.0, box2[1] + box2[3] / 2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
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


def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0].item() - boxes1[2].item() / 2.0, boxes2[0].item() - boxes2[2].item() / 2.0)
        Mx = torch.max(boxes1[0].item() + boxes1[2].item() / 2.0, boxes2[0].item() + boxes2[2].item() / 2.0)
        my = torch.min(boxes1[1].item() - boxes1[3].item() / 2.0, boxes2[1].item() - boxes2[3].item() / 2.0)
        My = torch.max(boxes1[1].item() + boxes1[3].item() / 2.0, boxes2[1].item() + boxes2[3].item() / 2.0)
        w1 = boxes1[2].item()
        h1 = boxes1[3].item()
        w2 = boxes2[2].item()
        h2 = boxes2[3].item()
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / (uarea+ 1e-12)

def Bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area =    torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                    torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def ensemble(boxes, en_thresh=0.0):
    if len(boxes) == 0:
        return None
    out_boxes = torch.zeros((3))
    cnt_valid = 0
    for i in range(len(boxes)):
        if len(boxes[i]) > 0:
            if boxes[i][0][0][2] > en_thresh:
                out_boxes[0] += boxes[i][0][0][0]
                out_boxes[1] += boxes[i][0][0][1]
                out_boxes[2] += boxes[i][0][0][2]
                cnt_valid += 1
    if cnt_valid == 0:
        print("prediction confidence smaller than en_thresh")
        out_boxes = torch. tensor([0, 0, 1])
        return out_boxes
    else:
        out_boxes /= cnt_valid
    return out_boxes


# def ensemble(boxes, en_thresh=0.0):
#     if len(boxes) == 0:
#         return None
#     out_boxes = torch.zeros((3))
#     cnt_valid = 0
#     for i in range(len(boxes)):
#         if boxes[i][2] > en_thresh:
#             out_boxes[0] += boxes[i][0]
#             out_boxes[1] += boxes[i][1]
#             out_boxes[2] += boxes[i][2]
#             cnt_valid += 1
#     if cnt_valid == 0:
#         print("prediction confidence smaller than en_thresh")
#         return None
#     else:
#         out_boxes /= cnt_valid
#     return out_boxes

def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


def get_region_boxes(output, conf_thresh, num_classes, only_objectness=1, validation=False, device=None):

    all_boxes = []
    batch = output.size(0)
    h = output.size(2)
    w = output.size(3)

    output = output.view(batch , 3, h * w).transpose(0, 1).contiguous().view(3, batch * h * w)

    grid_x = torch.linspace(0, w - 1, w).repeat(h, 1).repeat(batch, 1, 1).view(
        batch * h * w).to(device)  # type_as(output)
    grid_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().repeat(batch, 1, 1).view(
        batch * h * w).to(device)  # type_as(output)
    xs = torch.sigmoid(output[0].data) + grid_x
    ys = torch.sigmoid(output[1].data) + grid_y

    det_confs = torch.sigmoid(output[2].data)
    det_confs = det_confs.view(batch,  h * w).cpu()
    xs = xs.view(batch, h * w).cpu()
    ys = ys.view(batch, h * w).cpu()
    # temp_confs = det_confs[det_confs > conf_thresh]
    for i in range(batch):
        boxes = []
        index = np.arange(h * w)
        max_conf = torch.max(det_confs[i])
        if max_conf < conf_thresh:
            print("no kp found")
            break
        index = torch.from_numpy(index[det_confs[i] == max_conf])
        bcx = xs[i][index==1] / w
        bcy = ys[i][index==1] / h
        det_conf = det_confs[i][index == 1]
        for j in range(len(bcx)):
            box = [bcx[j], bcy[j],det_conf[j]]
            boxes.append(box)
        all_boxes.append(boxes)

    return all_boxes


def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    import cv2
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0],[1, 1, 0]]);

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(round((box[0] - box[2] / 2.0) * width))
        y1 = int(round((box[1] - box[3] / 2.0) * height))
        x2 = int(round((box[0] + box[2] / 2.0) * width))
        y2 = int(round((box[1] + box[3] / 2.0) * height))

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img


def plot_boxes(img, boxes, savename=None, class_names=None, GTfile = None):
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0.5],[0.5,1,0],[0,0.5,1]]);

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    if boxes is not None:
        for i in range(len(boxes)):
            box = boxes[i]
            x1 = (box[0] - box[2] / 2.0) * width
            y1 = (box[1] - box[3] / 2.0) * height
            x2 = (box[0] + box[2] / 2.0) * width
            y2 = (box[1] + box[3] / 2.0) * height

            rgb = (255, 0, 0)
            if len(box) >= 7 and class_names:
                cls_conf = box[5]
                cls_id = box[6]
                print('%s: %f' % (class_names[cls_id], cls_conf))
                classes = len(class_names)
                offset = cls_id * 123457 % classes
                red = get_color(2, offset, classes)
                green = get_color(1, offset, classes)
                blue = get_color(0, offset, classes)
                rgb = (red, green, blue)
                fo = ImageFont.truetype("consola.ttf", size=10)
                draw.text((x1, y1), class_names[cls_id], fill=rgb, font=fo)
                draw.text((x1, y1),str(cls_conf.numpy()), fill=rgb, font=fo)
            draw.rectangle([x1, y1, x2, y2], outline=rgb)

    if GTfile is not None:
        tmp = torch.from_numpy(read_truths_args(GTfile, 6.0 / img.width).astype('float32'))
        for i in range(tmp.size(0)):
            x1 = (tmp[i][1] - tmp[i][3] / 2.0) * width
            y1 = (tmp[i][2] - tmp[i][4] / 2.0) * height
            x2 = (tmp[i][1] + tmp[i][3] / 2.0) * width
            y2 = (tmp[i][2] + tmp[i][4] / 2.0) * height
            rgb = (0,0,0)
            draw.text((x1, y1), class_names[int(tmp[i][0])], fill=rgb)
            draw.rectangle([x1, y1, x2, y2], outline=rgb)
    if savename:
        print("save plot results to %s" % savename)
        img.save(savename)
    return img


def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(int(truths.size / 5), 5)  # to avoid single truth problem
        return truths
    else:
        return np.array([])


def read_truths_args(lab_path, min_box_scale):
    truths = read_truths(lab_path)
    new_truths = []
    for i in range(truths.shape[0]):
        if truths[i][3] < min_box_scale:
            continue
        new_truths.append([truths[i][0], truths[i][1], truths[i][2], truths[i][3], truths[i][4]])
    return np.array(new_truths)


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


def image2torch(img):
    width = img.width
    height = img.height
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
    img = img.view(1, 3, height, width)
    img = img.float().div(255.0)
    return img


def do_detect(model, img, conf_thresh, nms_thresh, device):
    model.eval()
    # transform_totensor = transforms.ToTensor()
    # img = transform_totensor(img)
    img = img.unsqueeze(0).to(device)
    all_boxes = []
    with torch.no_grad():
        out1, out2, out3 = model(img)
    Out = [out1, out2, out3]
    for i in range(len(Out)):
        all_boxes.append(get_region_boxes(Out[i], conf_thresh, 7, device=device))

    if len(all_boxes) > 0:
        # boxes = all_boxes[0][0] + all_boxes[1][0] + all_boxes[2][0]
        boxes = ensemble(all_boxes, en_thresh=0.0)
    else:
        boxes = torch.tensor([0,0,1])
    return boxes

def do_evaluate(pred_boxes, GTfile, width, height, num_classes):
    gt_boxes = torch.from_numpy(read_truths_args(GTfile, 6.0 / width).astype('float32'))
    num_GT_positive = np.zeros((num_classes), dtype = np.uint8)
    num_preds = np.zeros((num_classes), dtype = np.uint8)
    num_correct = np.zeros((num_classes), dtype = np.uint8)
    for j in range(len(pred_boxes)):
        num_preds[int(pred_boxes[j][-1])] += 1
    for i in range(gt_boxes.size(0)):
        num_GT_positive[int(gt_boxes[i][0])] += 1

    for i in range(gt_boxes.size(0)):
        for j in range(len(pred_boxes)):
            mx = min(pred_boxes[j][0] - pred_boxes[j][2] / 2.0, gt_boxes[i][1] - gt_boxes[i][3] / 2.0)
            Mx = max(pred_boxes[j][0] + pred_boxes[j][2] / 2.0, gt_boxes[i][1] + gt_boxes[i][3] / 2.0)
            my = min(pred_boxes[j][1] - pred_boxes[j][3] / 2.0, gt_boxes[i][2] - gt_boxes[i][4] / 2.0)
            My = max(pred_boxes[j][1] + pred_boxes[j][3] / 2.0, gt_boxes[i][2] + gt_boxes[i][4] / 2.0)
            w1 = pred_boxes[j][2]
            h1 = pred_boxes[j][3]
            w2 = gt_boxes[i][2]
            h2 = gt_boxes[i][3]
            uw = Mx - mx
            uh = My - my
            cw = w1 + w2 - uw
            ch = h1 + h2 - uh
            if cw <= 0 or ch <= 0:
                iou = 0.0
            else:
                area1 = w1 * h1
                area2 = w2 * h2
                carea = cw * ch
                uarea = area1 + area2 - carea
                iou = (carea / (uarea + 1e-12)).item()
                print(iou)
            if iou > 0.5 and gt_boxes[i][0] == pred_boxes[j][-1].float():
                num_correct[int(gt_boxes[i][0])] += 1
                break
    return num_preds, num_GT_positive, num_correct


def parse_config_info(configFile):
    infos = dict()
    with open(configFile, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if line == '' or line[0] == '#':
            continue
        key, value = line.split('=')
        key = key.strip()
        value = value.strip()
        infos[key] = value
    return infos


def scale_bboxes(bboxes, width, height):
    import copy
    dets = copy.deepcopy(bboxes)
    for i in range(len(dets)):
        dets[i][0] = dets[i][0] * width
        dets[i][1] = dets[i][1] * height
        dets[i][2] = dets[i][2] * width
        dets[i][3] = dets[i][3] * height
    return dets


def file_lines(thefilepath):
    count = 0
    thefile = open(thefilepath, 'r')
    while True:
        buffer = thefile.read(8192 * 1024)
        if not buffer:
            break
        count += buffer.count('\n')
    thefile.close()
    return count


def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg' or imghdr.what(fname) == 'jpg':
            try:
                fhandle.seek(0)  # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                    # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception:  # IGNORE:W0703
                return
        else:
            return
        return width, height


def logging(message):
    print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))

def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));     start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start+num_b]));  start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w
    return start

def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    conv_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w
    return start


def RoI_Align(inputs, rois, align_size):
    nB = inputs[0].size(0)
    output_feature = []
    for i in len(inputs):
        batch_boxes = rois[i][0]
        input = inputs[i]
        h = input.size(2)
        w = input.size(3)
        for b in nB:
            feature = input[b]
            boxes = batch_boxes[b]
            for box in boxes:
                x1 = np.floor((box[0] - box[2] / 2.0) * w)
                y1 = np.floor((box[1] - box[3] / 2.0) * h)
                x2 = np.ceil((box[0] + box[2] / 2.0) * w)
                y2 = np.ceil((box[1] + box[3] / 2.0) * h)
                output_feature.append(F.interpolate(feature[..., y1:y2, x1:x2], size=align_size, mode='bilinear'))

    return output_feature

# 模型显存占用监测函数
# model：输入的模型
# input：实际中需要输入的Tensor变量
# type_size 默认为 4 默认类型为 float32

def modelsize(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, torch.nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums

    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size * 2 / 1000 / 1000))
