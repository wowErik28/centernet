import torch
from torch.utils.data import DataLoader
from torchsummary import summary
import cv2

from utils.dataloader import CenternetDataset
from utils.utils import decode_box, nms, softnms, centernet_correct_box
from nets.centernet import CenternetResnet50
from centernet import Centernet
from nets.centernet_training import focal_loss, reg_l1_loss

'''
以下为数据预处理及导入的代码
'''
# with open('2007_train.txt', 'r', encoding='utf-8') as file:
#     lines = file.readlines()
#
# input_shape = (512,512,3)
# dataset = CenternetDataset(lines[:9], input_shape, n_classes=20)
# gen = DataLoader(dataset, batch_size=3)
# #
# img, batch_hm, batch_wh, batch_reg, batch_reg_mask = None, None, None, None, None
# for batch in gen:
#     img, batch_hm, batch_wh, batch_reg, batch_reg_mask = batch
#     print('img.shape', img.shape, img.type())
#     print('hm.shape', batch_hm.shape, batch_hm.type())
#     print('wh.shape', batch_wh.shape, batch_wh.type())
#     print('reg.shape', batch_reg.shape, batch_reg.type())
#     print('regmask.shape', batch_reg_mask.shape, batch_reg_mask.type())

'''
以下为 初始化网络的代码
'''

# model = CenternetResnet50(n_class=20, pretrain_path='model_data/resnet50.pth', pretrain=True)
# # summary(model.cuda(), [(3,512,512)], batch_size=3)
# model.freeze_backbone()
# model = model.cuda()
# print('model finish********************************')
# '''
# 以下为 测试decode box的代码
# '''
# pred_hms, pred_whs, pred_offsets = model(img.float().cuda())
# res = decode_box(pred_hms, pred_whs, pred_offsets, 0.918, cuda=True)
# nms_res = nms(res, 0.7)
#
# for i in range(len(res)):
#     if len(res[i]) == 0 or len(nms_res[i]) == 0:
#         print('[]')
#         continue
#     print('res',res[i].shape)
#     print('nms',nms_res[i].shape)
#     print('res_before', res[i])
#     print('nms_after',nms_res[i])
#     xm, ym, xa, ya, conf, type = nms_res[i][:,0], nms_res[i][:,1], nms_res[i][:,2], nms_res[i][:,3], \
#                                  nms_res[i][:,4], nms_res[i][:,5]
#     correct_res = centernet_correct_box(xm.reshape(-1,1), ym.reshape(-1,1), xa.reshape(-1,1), ya.reshape(-1,1),
#                                 [512, 512], [1330, 1330])
#     print('correct_res.shape', correct_res.shape)
#     print('correct_res',correct_res)

'''
测试主目录的 centernet.py
'''
centernet = Centernet()
img = cv2.imread(r'D:\Project\CV\yolo3_learning/VOCdevkit/VOC2007/JPEGImages/000007.jpg')
img = centernet.detect_image(img)
cv2.imshow('image', img)
cv2.waitKey()

'''
测试训练代码 
'''
epochs = 1
bsz = 3
n_class = 20

model = CenternetResnet50(n_class=n_class, pretrain_path='model_data/resnet50.pth', pretrain=True)
# summary(model.cuda(), [(3,512,512)], batch_size=3)
model.freeze_backbone()
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print('model finish********************************')


with open('2007_train.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
input_shape = (512,512,3)
dataset = CenternetDataset(lines[:100], input_shape, n_classes=n_class)
gen = DataLoader(dataset, batch_size=bsz)


model = model.train()
for epoch in range(epochs):
    print('Epoch : {} training'.format(epoch))
    for batch in gen:
        img, batch_hm, batch_wh, batch_reg, batch_reg_mask = batch


        pred_hms, pred_whs, pred_offsets = model(img.cuda())
        focal_loss1 = focal_loss(pred_hms, batch_hm.cuda())
        l1_loss_offset = reg_l1_loss(pred_offsets, batch_reg.cuda(), batch_reg_mask.cuda())
        l1_loss_wh = reg_l1_loss(pred_whs, batch_wh.cuda(), batch_reg_mask.cuda())
        loss = focal_loss1 + l1_loss_wh + l1_loss_offset

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
torch.save(model.state_dict(), 'model_data/resnet50_centernet.pth')
