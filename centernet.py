import os

import cv2
import torch
import numpy as np

from nets.centernet import CenternetResnet50
from utils.dataloader import preprocess_image
from utils.utils import decode_box, centernet_correct_box, nms

class Centernet:
    config = {
        "model_path": 'model_data/resnet50_centernet.pth',
        "classes_path": 'model_data/voc_classes.txt',
        "get_whole_model" : True,
        "backbone": 'resnet50',
        "backbone_path": 'model_data/resnet50.pth',

        "image_size": [512, 512, 3], #(h,w,c)
        "confidence": 0.3,

        "nms": True,
        "nms_threshold": 0.3,
        "cuda": True
    }
    @classmethod
    def get(cls, key):

        return cls.config[key]

    def __init__(self):

        self.class_names = self._get_class()

        self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(Centernet.get('classes_path'))
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def generate(self):
        print('Loading weights into state dict...')
        model = CenternetResnet50(n_class=len(self.class_names), pretrain_path=Centernet.get('backbone_path'), pretrain=False)


        if Centernet.get('get_whole_model'):
            print('load whole model')
            state_dict = torch.load(Centernet.get('model_path'))
            model.load_state_dict(state_dict)
        if Centernet.get('cuda'):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.cuda()
        else:
            self.device = torch.device('cpu')

        self.centernet = model
        print('Model loading finish')

    def detect_image(self, img):
        '''
        :param img: ndarray (h,w,c)
        :return: img: ndarray (h,w,c) 已经将框画好了
        '''
        #首先对图片进行预处理
        img_ = preprocess_image(img)
        ih, iw, c = img.shape
        img_tensor = torch.tensor(img_).to(self.device).view(1, ih, iw, c).permute((0,3,1,2)).contiguous().float()

        with torch.no_grad():
            pred_hms, pred_whs, pred_offsets  = self.centernet(img_tensor)    #(1,20,128,128), (1,2,128,128), (1,2,128,128)

            outputs = decode_box(pred_hms, pred_whs, pred_offsets, Centernet.get('confidence'),
                                 cuda=Centernet.get('cuda')) #(bsz,*,4+1+1) or [[]]

            if Centernet.get('nms'):
                outputs = nms(outputs, Centernet.get('nms_threshold'))  #(bsz,n,4+1+1)or [[]]

            #没有框检测到
            if len(outputs[0]) == 0:
                print('No detection box')
                return img

            outputs = outputs[0] #(n,4+1+1)
            pred_conf = outputs[:,-2]
            pred_classes = outputs[:, -1].astype('int32')
            correct_outputs = centernet_correct_box(outputs[:, 1], outputs[:, 0], outputs[:, 3], outputs[:, 2],
                                  Centernet.get('image_size')[:-1], (ih, iw))    #(n,4)

        for i in range(len(correct_outputs)):

            xm, ym, xa, ya = correct_outputs[i]
            xm = xm - 5
            ym = ym - 5
            xa = xa + 5
            ya = ya + 5

            xm = max(0, np.floor(xm + 0.5).astype('int32'))
            ym = max(0, np.floor(ym + 0.5).astype('int32'))
            xa = min(iw, np.floor(xa + 0.5).astype('int32'))
            ya = min(ih, np.floor(ya + 0.5).astype('int32'))
            if xm >= xa or ym >= ya:
                continue
            img = cv2.rectangle(img, (xm, ym), (xa, ya), color=(0,0,255))
            print(pred_conf[i])
            print(pred_classes[i])
        return img
