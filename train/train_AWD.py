import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.AWD.yolo import YoloBody
from nets.yolo_training import (YOLOLoss, get_lr_scheduler, set_optimizer_lr,
                                weights_init)
from nets.AWD.shallow_darknet import ShallowNet
from utils.AFB.loss import AFBLoss
from utils.callbacks import LossHistory
from utils.AWD_dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_classes
from utils.fit import fit_one_epoch

if __name__ == "__main__":
    Cuda = True
    classes_path = 'model_data/classes.txt'
    model_path = 'model_data/yolox_s.pth'
    SPT_model_path = 'model_data/SPT.pth'
    IRT_model_path = 'model_data/IRT.pth'

    input_shape = [640, 640]
    phi = 's'
    mosaic = False

    Init_Epoch = 0
    Freeze_Epoch = 0
    Freeze_batch_size = 8

    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 8

    Freeze_Train = False

    Init_lr = 1e-2
    Min_lr = Init_lr * 0.01

    optimizer_type = "sgd"
    momentum = 0.937
    weight_decay = 5e-4

    lr_decay_type = "cos"

    save_period = 50

    num_workers = 4

    train_annotation_path = 'datasets/data_info/train_fog.txt'
    val_annotation_path = 'datasets/data_info/val_fog.txt'
    clear_annotation_path = 'datasets/data_info/train_clear.txt'
    val_clear_annotation_path = 'datasets/data_info/val_clear.txt'

    class_names, num_classes = get_classes(classes_path)

    model = YoloBody(num_classes, phi)
    weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        if "model" in pretrained_dict:
            pretrained_dict = pretrained_dict["model"]
        #for k, v in pretrained_dict.items():
            #print(k)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    yolo_loss = YOLOLoss(num_classes)
    loss_history = LossHistory("logs/", model, input_shape=input_shape)
    model_train = model.train()
    if Cuda:
        cudnn.benchmark = True
        model_train = model_train.cuda()


    'IRT'
    res_model = ShallowNet()
    weights_init(res_model)
    print('Load restoration branch weights {}.'.format(IRT_model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res_model_dict = res_model.state_dict()
    res_pretrained_dict = torch.load(IRT_model_path, map_location=device)
    res_pretrained_dict_filtered = {}
    for k, v in res_pretrained_dict.items():
        if k in res_model_dict and res_model_dict[k].shape == v.shape:  # 注意使用.shape而不是np.shape()
            res_pretrained_dict_filtered[k] = v
    res_model_dict.update(res_pretrained_dict_filtered)
    res_model.load_state_dict(res_model_dict)
    res_model = res_model.eval()
    for (name, param) in res_model.named_parameters():
        param.requires_grad = False
        #print(name)
    if Cuda:
        res_model = res_model.cuda()

    'SPT'
    det_model = ShallowNet()
    weights_init(det_model)
    print('Load detection branch weights {}.'.format(SPT_model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    det_model_dict = det_model.state_dict()
    det_pretrained_dict = torch.load(SPT_model_path, map_location=device)
    det_pretrained_dict_filtered = {}
    for k, v in det_pretrained_dict.items():
        if k in det_model_dict and det_model_dict[k].shape == v.shape:  # 注意使用.shape而不是np.shape()
            det_pretrained_dict_filtered[k] = v
    det_model_dict.update(det_pretrained_dict_filtered)
    det_model.load_state_dict(det_model_dict)
    det_model = det_model.eval()
    for (name, param) in det_model.named_parameters():
        param.requires_grad = False
        #print(name)
    if Cuda:
        det_model = det_model.cuda()

    'D_guidance'
    D_guidance = AFBLoss(alpha_mgd=0.0000001)
    weights_init(D_guidance)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    D_guidance = D_guidance.train()
    if Cuda:
        D_guidance = D_guidance.cuda()
    print('Detection Guidance Down')
    'R_guidance'
    R_guidance = AFBLoss(alpha_mgd=0.0000001)
    weights_init(R_guidance)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    R_guidance = R_guidance.train()
    if Cuda:
        R_guidance = R_guidance.cuda()


    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    with open(clear_annotation_path, encoding='utf-8') as f:
        clear_lines = f.readlines()
    with open(val_clear_annotation_path, encoding='utf-8') as f:
        val_clear_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if True:
        UnFreeze_flag = False

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs = 64
        Init_lr_fit = max(batch_size / nbs * Init_lr, 1e-4)
        Min_lr_fit = max(batch_size / nbs * Min_lr, 1e-6)

        pg0, pg1, pg2 = [], [], []
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)
        optimizer = {
            'adam': optim.Adam(pg0, Init_lr_fit, betas=(momentum, 0.999)),
            'sgd': optim.SGD(pg0, Init_lr_fit, momentum=momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Dataset error!")

        train_dataset = YoloDataset(train_lines, clear_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch,
                                    mosaic=mosaic, train=True)
        val_dataset = YoloDataset(val_lines, val_clear_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch,
                                  mosaic=False, train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        for epoch in range(Init_Epoch, UnFreeze_Epoch):

            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs = 64
                Init_lr_fit = max(batch_size / nbs * Init_lr, 1e-4)
                Min_lr_fit = max(batch_size / nbs * Min_lr, 1e-6)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("Dataset error！")

                gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=yolo_dataset_collate)
                gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=yolo_dataset_collate)

                UnFreeze_flag = True

            gen.dataset.epoch_now = epoch
            gen_val.dataset.epoch_now = epoch

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val,
                          gen, gen_val, UnFreeze_Epoch, Cuda, save_period, res_model, det_model,R_guidance , D_guidance,
                          )