import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

import cv2
# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn
def save_roi(image_name,pred,dir):
    image = cv2.imread(image_name)
    predict = pred.squeeze()
    predict = predict.cpu().data.numpy()
    predict = cv2.resize(predict, (image.shape[1], image.shape[0]))
    #cv2.threshold(predict, 0.001, 1.0, cv2.THRESH_BINARY, predict)
    #kernel = np.ones((10, 10), np.uint8)
    #cv2.dilate(predict, kernel, predict, iterations=3)
    predict = np.tile(predict[:, :, np.newaxis], (1, 1, 3))
    image = np.multiply(image, predict)
    # cv2.imshow("m",predict)
    # cv2.imshow("img",image)
    # cv2.waitKey()
    img_name = image_name.split("/")[-1]
    cv2.imwrite(dir + '/' + img_name, image)
def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')


def roi_detect(data_dir,roi_dir):
    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp

    image_dir = os.path.join(os.getcwd(), data_dir)
    prediction_dir = os.path.join(os.getcwd(), roi_dir)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')

    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_roi(img_name_list[i_test],pred,prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7

def video2imgs(video_path,save_dir):
    video_in = cv2.VideoCapture(video_path)
    frame_id = 0
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    while True:
        ok, frame = video_in.read()
        if not ok:
            break
        save_name = os.path.join(save_dir,"%04d" % frame_id+'.jpg')
        cv2.imwrite(save_name,frame)
        frame_id+=1
def imgs2video(roi_dir):
    img_name_list = glob.glob(roi_dir + os.sep + '*')
    img_name_list.sort()
    frame_list = []
    for img_name in img_name_list:
        img = cv2.imread(img_name)
        mask = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.threshold(mask, 0.001, 255, cv2.THRESH_BINARY,mask)
        kernel = np.ones((10, 10), np.uint8)
        cv2.erode(mask, kernel, mask, iterations=3)
        cv2.dilate(mask, kernel, mask, iterations=3)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # cv2.RETR_EXTERNAL 定义只检测外围轮廓
        cnts = contours[0]
        center_x = int(img.shape[1]/2)
        center_y = int(img.shape[0]/2)
        for cnt in cnts:
            # 外接矩形框，没有方向角
            x, y, w, h = cv2.boundingRect(cnt)
            if x+w >= img.shape[1] or y+h >= img.shape[0]:
                continue
            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame = np.zeros_like(img)
            frame[center_y-h//2:center_y+h//2,center_x-w//2:center_x+w//2] = img[y:y+(h//2)*2, x:x+(w//2)*2]
            frame_list.append(frame)
            #
            # # 最小外接矩形框，有方向角
            # rect = cv2.minAreaRect(cnt)
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
            #
            # # 最小外接圆
            # (x, y), radius = cv2.minEnclosingCircle(cnt)
            # center = (int(x), int(y))
            # radius = int(radius)
            # cv2.circle(img, center, radius, (255, 0, 0), 2)
            #
            # # 椭圆拟合
            # ellipse = cv2.fitEllipse(cnt)
            # cv2.ellipse(img, ellipse, (255, 255, 0), 2)
            #
            # # 直线拟合
            # rows, cols = img.shape[:2]
            # [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            # lefty = int((-x * vy / vx) + y)
            # righty = int(((cols - x) * vy / vx) + y)
            # img = cv2.line(img, (cols - 1, righty), (0, lefty), (0, 255, 255), 2)
            #cv2.imshow('a', frame)
            #cv2.waitKey(10)
    cv2.namedWindow('3D')
    cv2.waitKey(10)

    def show_frame(x):
        cv2.imshow('3D',frame_list[x])

    cv2.imshow('3D', frame_list[0])
    cv2.createTrackbar('angle', '3D', 0, len(frame_list)-1,show_frame)
    while True:
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    video_path='test_data/beizi.mp4'
    imgs_dir='test_data/beizi'
    roi_dir='test_data/beizi_roi'
    # video2imgs(video_path,imgs_dir)
    # roi_detect(imgs_dir,roi_dir)
    imgs2video(roi_dir)