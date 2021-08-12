import math
from random import shuffle

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

from .utils import gaussian_radius, draw_gaussian

def preprocess_image(image):
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return ((np.float32(image) / 255.) - mean) / std

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class CenternetDataset(Dataset):

    def __init__(self, train_lines, input_shape, n_classes
                 , is_train = True):
        '''
        train_lines: [‘line’, ‘line’ , ‘line’,… ]
        input_shape: 网络输入的w,h,c
        Is_train: True or False
        output_shape = 128,128,3
        '''
        super().__init__()
        self.train_lines = train_lines
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.is_train = is_train

        self.output_shape = (int(input_shape[0]/4),int(input_shape[1]/4),
                             int(input_shape[2]))

    def get_random_data(self, line, input_shape):
        #process data
        '''
        #可共享代码 与网络类型不耦合
        line: ‘000005.jpg 263,211,324,339,8 165,264,253,372,8’
        input_shape:网络输入的w,h,c

        return img:ndarray, target:ndarray
        Img:ndarray   shape(h,w,c)
        target:[[xm,ym,xa,ya,class],[xm,ym,xa,ya,class]] shape(*,5)
        '''
        string_list = line.strip().split()
        box_array = np.array([list(map(int, box.split(',')))
                              for box in string_list[1:]]).reshape(-1,5) #(*,5)
        img = cv2.imread(string_list[0])
        ih, iw, ic = img.shape
        img = cv2.resize(img, (input_shape[0],input_shape[1]))

        box_array[:, [1, 3]] = box_array[:, [1, 3]] / ih * input_shape[1]
        box_array[:, [0, 2]] = box_array[:, [0, 2]] / iw * input_shape[0]
        # change the bbox's cordinate
        if self.is_train:
            pass
        return img, box_array

    def __getitem__(self, index):
        '''
        __getitem__(self, index):->img:ndarray, target:ndarray
        #需要调用 get_random_data
        input
            index:0~len(self)
        return
            all is ndarray
            img.shape (c,h,w)                       np.float32
            batch_hm(out_h,out_w,n_classes) #热力图  np.float32
            batch_wh(out_h,out_w, 2)#宽高            np.float32
            batch_reg(out_h,out_w, 2)#中心点         np.float32
            batch_reg_mask(out_h,out_w) )#中心点存在的地方为1 np.float32
        '''
        if index == 0:
            shuffle(self.train_lines)

        line = self.train_lines[index]
        img, box_array = self.get_random_data(line, self.input_shape)
        img = preprocess_image(img)
        img = np.transpose(img, (2, 1, 0)).astype('float32')

        batch_hm = np.zeros((self.output_shape[1], self.output_shape[0],self.n_classes),
                            dtype = np.float32)
        batch_wh = np.zeros((self.output_shape[1], self.output_shape[0], 2),
                            dtype = np.float32)
        batch_reg = np.zeros((self.output_shape[1], self.output_shape[0], 2),
                            dtype=np.float32)
        batch_reg_mask = np.zeros((self.output_shape[1], self.output_shape[0]),
                             dtype=np.float32)

        if len(box_array) != 0:
            # change box shape to adapt to output_shape
            box_array[:, [1, 3]] = box_array[:, [1, 3]] / self.input_shape[1] * self.output_shape[1]

            box_array[:, [0, 2]] = box_array[:, [0, 2]] / self.input_shape[0] * self.output_shape[0]

        for i in range(len(box_array)):
            bbox = box_array[i].copy()
            cls = bbox[4]
            cy = int((bbox[3] + bbox[1]) / 2)
            cx = int((bbox[2] + bbox[0]) / 2)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                cls_id = int(box_array[i, -1])
                batch_hm[:, :, cls_id] = draw_gaussian(batch_hm[:, :, cls_id], (cx, cy), radius)
                batch_hm[:,:,cls] = 1
                batch_wh[cy,cx,:] = w, h
                batch_reg[cy,cx,:] = (bbox[3] + bbox[1]) / 2 - cy, (bbox[2] + bbox[0]) / 2 - cx
                batch_reg_mask[cy,cx] = 1

        batch_hm = np.transpose(batch_hm, (2, 1, 0))
        batch_wh = np.transpose(batch_wh, (2, 1, 0))
        batch_reg = np.transpose(batch_reg, (2, 1, 0))

        return img, batch_hm, batch_wh, batch_reg, batch_reg_mask

    def __len__(self):
        return len(self.train_lines)

