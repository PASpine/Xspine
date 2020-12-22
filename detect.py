import sys
import time
from PIL import Image, ImageDraw
import torch
import cv2
from utils import *
from model import *
from config import Config
import pandas as pd

def detect(weightfile, yolo_config, imgDir, outDir=None, GTfile=None, gpu='0'):

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("GPU is available!")
    else:
        print("GPU is not available!!!")
    device = torch.device("cuda:" + gpu if use_cuda else "cpu")

    model = YOLO(yolo_config.num_classes, yolo_config.in_channels)

    model.load_weights(weightfile)
    model.height = 384
    model.width = 384

    model = model.to(device)
    imgList = os.listdir(imgDir)
    count = 0
    Fovea_loc = np.zeros((400, 2))
    for i in range(len(imgList)):
        line =imgList[i]
        print(count, line)
        image = np.float32(cv2.imread(os.path.join(imgDir, line)))
        height_Y, width_X, channel_X = image.shape
        image_t = cv2.resize(image, (yolo_config.init_height, yolo_config.init_width), interpolation=cv2.INTER_LINEAR)
        image_t = image_t / 255.0
        image_t = np.transpose(image_t, (2, 0, 1))
        image_t = torch.tensor(image_t).float()
        image_t = image_t.to(device)

        boxes = do_detect(model, image_t, 0.5, 0.4, device)
        # if boxes is None:
        #     print('No boxes!!!!')
        boxes = np.array(boxes.cpu())
        savename = os.path.join(outDir, line)
        cv2.circle(image, (int(boxes[0]*width_X), int(boxes[1]*height_Y)), 5, (0,255,0), -1)
        cv2.imwrite(savename, image)
        Fovea_loc[i, 0] = boxes[0] * width_X
        Fovea_loc[i, 1] = boxes[1] * height_Y


        count += 1


    dict = {'FileName': imgList, 'Fovea_X': Fovea_loc[:, 0], 'Fovea_Y': Fovea_loc[:, 1]}
    df = pd.DataFrame(dict)
    df.columns = ['FileName', 'Fovea_X', 'Fovea_Y']
    df.to_csv('/media/glo/新加卷/Data/AMD/valid_yolo_channel3.csv')
    print('Done!')

if __name__ == '__main__':
    GTfile = "none"
    weightfile = "/home/glo/Desktop/Projects/YOLO_XSL/backup/000250.pkl"
    # weightfile = "/home/glo/Desktop/Projects/YOLO_XSL/backup_condition/000650_old.pkl"
    imgDir = "/media/glo/新加卷/Data/AMD/Validation-400-images"#"/media/glo/新加卷/Data/AMD/Validation-400-images"  "/media/glo/新加卷/Data/AMD/Training400/Image"
    outDir = "/media/glo/新加卷/Data/AMD/pre_valid_regress"
    yolo_config = Config()
    detect(weightfile, yolo_config, imgDir, outDir, GTfile, '0')
