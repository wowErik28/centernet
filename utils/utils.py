'''
用于decode 网络预测结果
'''
import torch
import torch.nn.functional as F
import numpy as np

def centernet_correct_box(ym, xm, ya, xa, input_shape, image_shape):
    '''
    数据预处理时是将image_shape resize 成 input_shape
    现在ym xm ya xa 都是0-1比例信息，只要*image_shape即可
    :param ym:(n,1)
    :param xm:(n,1)
    :param ya:(n,1)
    :param xa:(n,1)
    :param input_shape:[512,512]
    :param image_shape:[1300,1300]
    :return: ndarray float (n,4)
    '''
    xm *= image_shape[1]
    xa *= image_shape[1]
    ym *= image_shape[0]
    ya *= image_shape[0]
    return np.concatenate((xm.reshape(-1, 1), ym.reshape(-1, 1), xa.reshape(-1, 1), ya.reshape(-1, 1)),
                          axis=-1)


def iou(b1, b2):
    '''
    必须是xm,ym,xa,ya形式
    :param b1: ndarray(4)
    :param b2: ndarray(bsz, 4)
    :return: float score_array (bsz)
    '''
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:,0], b2[:,1], b2[:,2], b2[:,3]

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.maximum((inter_x2 - inter_x1) * (inter_y2 - inter_y1), 0)
    uion_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) + (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area/np.maximum(uion_area-inter_area, 1e-6)

def nms(bboxes, iou_threshold):
    '''
    :param bboxes: list [[bbox_array, bbox_array, ...],.....]   (bsz, *, 4+1+1)
    :param iou_threshold: float
    :return: (bsz,n,4+1+1)
    '''
    output_bboxes = []
    for i in range(len(bboxes)):
        bboxes_array = bboxes[i] #(*, 4+1+1)

        if len(bboxes_array) <= 1:
            output_bboxes.append(bboxes_array)
            continue

        class_array = np.unique(bboxes_array[:, -1])

        best_box_list = [] # (n, 4+1+1)
        for c in class_array:
            class_mask = (bboxes_array[:, -1] == c)
            class_boxes_array = bboxes_array[class_mask] #(gc, 4+1+1)

            if len(class_boxes_array) <= 0:
                continue
            elif len(class_boxes_array) == 1:
                best_box_list.append(class_boxes_array[0])
                continue

            argsort = np.argsort(class_boxes_array[:, -2])[::-1] #(*)
            class_boxes_array = class_boxes_array[argsort]

            #取出和best_box_list中iou<iou_threshold的那些bbox_array

            while len(class_boxes_array) !=0 :
                best_box_list.append(class_boxes_array[0])
                if len(class_boxes_array) == 1:
                    break
                ious = iou(best_box_list[-1], class_boxes_array[1:])
                class_boxes_array = class_boxes_array[1:][ious < iou_threshold]

        output_bboxes.append(np.array(best_box_list)) #(bsz,n,4+1+1)
    return output_bboxes

def softnms(bboxes, iou_threshold, sigma=0.5):
    '''
    :param bboxes: list [[bbox_array, bbox_array, ...],.....]   (bsz, *, 4+1+1)
    :param iou_threshold:
    :return: (bsz,n,4+1+1)
    '''
    output_bboxes = []
    for i in range(len(bboxes)):
        bboxes_array = bboxes[i] #(*, 4+1+1)

        if len(bboxes_array) <= 1:
            output_bboxes.append(bboxes_array)
            continue

        class_array = np.unique(bboxes_array[:, -1])

        best_box_list = [] # (n, 4+1+1)
        for c in class_array:
            class_mask = (bboxes_array[:, -1] == c)
            class_boxes_array = bboxes_array[class_mask] #(gc, 4+1+1)

            if len(class_boxes_array) <= 0:
                continue
            elif len(class_boxes_array) == 1:
                best_box_list.append(class_boxes_array[0])
                continue

            argsort = np.argsort(class_boxes_array[:, -2])[::-1] #(*)
            class_boxes_array = class_boxes_array[argsort]

            #取出和best_box_list中iou<iou_threshold的那些bbox_array

            while len(class_boxes_array) !=0 :
                best_box_list.append(class_boxes_array[0])
                if len(class_boxes_array) == 1:
                    break
                ious = iou(best_box_list[-1], class_boxes_array[1:])
                class_boxes_array[1:, 4] = np.exp(-(ious*ious)/sigma)*class_boxes_array[1:, 4]
                class_boxes_array = class_boxes_array[1:]
                score_mask = np.argsort(class_boxes_array[:,4])[::-1]
                class_boxes_array = class_boxes_array[score_mask]


        output_bboxes.append(np.array(best_box_list)) #(bsz,n,4+1+1)
    return output_bboxes

def pool_nms(hm_tensor, kernel_size=3):
    padding = (kernel_size-1)//2
    hmmax = F.max_pool2d(hm_tensor,(kernel_size, kernel_size), stride=1, padding=padding)
    mask = (hmmax == hm_tensor).float()
    return mask * hm_tensor

def decode_box(pred_hms, pred_whs, pred_offsets, threshold, cuda, topk=100):
    '''
    #decode batch by batch
    #in each batch: pool_nms;找到每个像素点对应的类别pred_conf和pred_type;找到大于threshold的 mask;
                    对于每一个mask有值点,得到 bbox:xm,ym,xa,ya,cls_prob,cls_type
    :param pred_hm: shape(bsz,n_class,128,128)
    :param pred_wh: shape(bsz,2,128,128)
    :param pred_offset: shape(bsz,2,128,128)
    :param threshold: float
    :param cuda: pred_hm等是否是在cuda上
    :return: np.array [[xm/128,ym/128,xa/128,ya/128,cls_prob,cls_type],.....]
    '''
    pred_hms = pool_nms(pred_hms) #(bsz, c,128,128)
    pred_hms = pred_hms.permute(0, 2, 3, 1).contiguous() #(bsz, 128,128, c)

    bsz, output_h, output_w, c = pred_hms.shape
    detect_lists = []
    for i in range(bsz):

        pred_hm = pred_hms[i].view(-1, c)         #(128*128, c)
        pred_wh = pred_whs[i].view(-1, 2)         #(128*128, 2)
        pred_offset = pred_offsets[i].view(-1, 2) #(128*128, 2)

        pred_conf, pred_class = torch.max(pred_hm, dim=-1)#(128*128),(128*128)
        mask = pred_conf > threshold  #(128*128)
        pred_wh = pred_wh[mask]    #(gt,2)
        pred_offset = pred_offset[mask] #(gt,2)
        pred_conf = pred_conf[mask]  #(gt)
        pred_class = pred_class[mask]#(gt)

        if len(pred_wh) == 0:
            detect_lists.append([])
            continue

        yv, xv = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))
        #(128*128),(128*128)
        yv = yv.flatten().float()[mask]#(gt)
        xv = xv.flatten().float()[mask]#(gt)

        if cuda:
            xv = xv.cuda()
            yv = yv.cuda()

        pred_centerx = xv + pred_offset[:, 0]
        pred_centery = yv + pred_offset[:, 1]
        pred_xm = pred_centerx - (pred_wh[:, 0]) / 2
        pred_xa = pred_centerx + (pred_wh[:, 0]) / 2
        pred_ym = pred_centery - (pred_wh[:, 1]) / 2
        pred_ya = pred_centery + (pred_wh[:, 1]) / 2

        detect = torch.cat([pred_xm.view(-1,1), pred_ym.view(-1,1),
                          pred_xa.view(-1,1), pred_ya.view(-1,1),
                          pred_conf.view(-1,1), pred_class.view(-1,1).float()], dim=1)
        detect[:, [0, 2]] = detect[:, [0, 2]]/output_w
        detect[:, [1, 3]] = detect[:, [1, 3]]/output_h
        #(gt, 6)
        argsort = torch.argsort(detect[:,-2], descending=True)
        detect_lists.append(detect[argsort][:topk].cpu().detach().numpy())

    return detect_lists

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

if __name__ == '__main__':
    b1 = np.array([1, 1, 3, 3])
    b2 = np.array([[2, 2, 4, 4]])
    res = iou(b1, b2)
    print(res)

