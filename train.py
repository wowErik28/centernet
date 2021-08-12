import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from utils.dataloader import CenternetDataset
from utils.utils import decode_box, nms, softnms, centernet_correct_box
from nets.centernet import CenternetResnet50
from nets.centernet_training import focal_loss, reg_l1_loss

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(net, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda,
                  optimizer, lr_scheduler):
    total_r_loss = 0
    total_c_loss = 0
    total_loss = 0
    val_loss = 0

    net.train()
    print('Start Train')
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            with torch.no_grad():
                if cuda:
                    batch = [ann.cuda() for ann in batch]

            batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

            optimizer.zero_grad()


            hm, wh, offset = net(batch_images)
            c_loss = focal_loss(hm, batch_hms)
            wh_loss = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
            off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)

            loss = c_loss + wh_loss + off_loss

            total_loss += loss.item()
            total_c_loss += c_loss.item()
            total_r_loss += wh_loss.item() + off_loss.item()

            loss.backward()
            optimizer.step()

            pbar.set_postfix(**{'total_r_loss': total_r_loss / (iteration + 1),
                                'total_c_loss': total_c_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            with torch.no_grad():
                if cuda:
                    batch = [ann.cuda() for ann in batch]


                batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch


                hm, wh, offset = net(batch_images)
                c_loss = focal_loss(hm, batch_hms)
                wh_loss = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                loss = c_loss + wh_loss + off_loss

                val_loss += loss.item()


            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)


    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
    print('Saving state, iter:', str(epoch + 1))
    torch.save(net.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
    (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    if lr_scheduler is not None:
        lr_scheduler.step(val_loss / (epoch_size_val + 1))
    return None


if __name__ == '__main__':
    input_shape = (512, 512, 3)
    classes_path = 'model_data/voc_classes.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)

    cuda = True
    lr = 1e-3
    bsz = 3
    Init_Epoch = 0
    Freeze_Epoch = 2
    Unfreeze_Epoch = 4
    #模型准备
    whole_model_path = 'model_data/resnet50_centernet.pth'
    print('Loading model.........................')
    model = CenternetResnet50(num_classes,None,False)
    state_dict = torch.load(whole_model_path)
    model.load_state_dict(state_dict)
    if cuda:
        model = model.cuda()
    print('Model finish..........................')

    #数据集准备
    annotation_path = '2007_train.txt'
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    lines = lines[: 40]
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    train_set = CenternetDataset(lines[: num_train], input_shape, n_classes=num_classes, is_train=True)
    val_set = CenternetDataset(lines[num_train: ], input_shape, n_classes=num_classes, is_train=False)
    gen = DataLoader(train_set, batch_size=bsz)
    gen_val = DataLoader(val_set, batch_size=bsz)
    epoch_size = num_train // bsz
    epoch_size_val = num_val // bsz
    if epoch_size == 0 or epoch_size_val == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

    #训练优化器等准备
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    '''
    这是未解冻的训练
    '''
    model.freeze_backbone()
    for epoch in range(Init_Epoch, Freeze_Epoch):
        fit_one_epoch(model, epoch=0, epoch_size=epoch_size, epoch_size_val=epoch_size_val,
                      gen=gen, genval=gen_val,Epoch=Freeze_Epoch, cuda=cuda,
                      optimizer=optimizer, lr_scheduler=lr_scheduler)

    '''
    这是解冻训练
    '''
    lr = 1e-4
    bsz = 4

    train_set = CenternetDataset(lines[: num_train], input_shape, n_classes=num_classes, is_train=True)
    val_set = CenternetDataset(lines[num_train:], input_shape, n_classes=num_classes, is_train=False)
    gen = DataLoader(train_set, batch_size=bsz)
    gen_val = DataLoader(val_set, batch_size=bsz)
    epoch_size = num_train // bsz
    epoch_size_val = num_val // bsz
    if epoch_size == 0 or epoch_size_val == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    model.unfreeze_backbone()
    for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
        fit_one_epoch(model, epoch=0, epoch_size=epoch_size, epoch_size_val=epoch_size_val,
                      gen=gen, genval=gen_val,Epoch=Freeze_Epoch, cuda=cuda,
                      optimizer=optimizer, lr_scheduler=lr_scheduler)