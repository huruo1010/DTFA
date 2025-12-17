import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.detection.yolo import YoloBody
from nets.detection.yolo_training import (YOLOLoss, get_lr_scheduler, set_optimizer_lr,
                                weights_init)
from utils.SPT.callbacks import LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_classes
from utils.SPT.utils_fit import fit_one_epoch


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Training Script')
    
    parser.add_argument('--cuda', action='store_true', default=True, help='Use CUDA')
    parser.add_argument('--classes_path', type=str, default='model_data/classes.txt', help='Path to classes file')
    parser.add_argument('--model_path', type=str, default='model_data/yolox_s.pth', help='Pretrained model path')
    parser.add_argument('--input_shape', type=int, nargs=2, default=[640, 640], help='Input image size [height, width]')
    parser.add_argument('--phi', type=str, default='s', choices=['s', 'm', 'l', 'x'], help='Model size')
    parser.add_argument('--mosaic', action='store_true', help='Use mosaic augmentation')
    
    parser.add_argument('--init_epoch', type=int, default=0, help='Initial epoch')
    parser.add_argument('--freeze_epoch', type=int, default=0, help='Freeze training epochs')
    parser.add_argument('--unfreeze_epoch', type=int, default=100, help='Unfreeze training epochs')
    parser.add_argument('--freeze_train', action='store_true', help='Enable freeze training')
    
    parser.add_argument('--freeze_batch_size', type=int, default=8, help='Batch size during freeze training')
    parser.add_argument('--unfreeze_batch_size', type=int, default=8, help='Batch size during unfreeze training')
    
    parser.add_argument('--init_lr', type=float, default=1e-2, help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-4, help='Minimum learning rate')
    parser.add_argument('--optimizer_type', type=str, default='sgd', choices=['sgd', 'adam'], help='Optimizer type')
    parser.add_argument('--momentum', type=float, default=0.937, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_decay_type', type=str, default='cos', choices=['cos', 'step', 'linear'], help='LR decay type')
    
    parser.add_argument('--train_annotation_path', type=str, default='datasets/data_info/train_clear.txt', help='Train annotation path')
    parser.add_argument('--val_annotation_path', type=str, default='datasets/data_info/val_clear.txt', help='Validation annotation path')
    
    parser.add_argument('--save_period', type=int, default=1, help='Save model every n epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--logs_dir', type=str, default='logs/SPT/', help='Logs directory')
    
    return parser.parse_args()


def load_pretrained_weights(model, model_path):
    if model_path != '':
        print(f'Load weights {model_path}.')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        
        if "model" in pretrained_dict:
            pretrained_dict = pretrained_dict["model"]
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and np.shape(model_dict[k]) == np.shape(v)}
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    return model


def setup_optimizer(model, args, batch_size, init_lr, min_lr):
    nbs = 64
    Init_lr_fit = max(batch_size / nbs * init_lr, 1e-4)
    Min_lr_fit = max(batch_size / nbs * min_lr, 1e-6)
    
    pg0, pg1, pg2 = [], [], []  
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
    
    if args.optimizer_type == 'adam':
        optimizer = optim.Adam(pg0, Init_lr_fit, betas=(args.momentum, 0.999))
    else:
        optimizer = optim.SGD(pg0, Init_lr_fit, momentum=args.momentum, nesterov=True)
    
    optimizer.add_param_group({"params": pg1, "weight_decay": args.weight_decay})
    optimizer.add_param_group({"params": pg2})
    
    lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, Init_lr_fit, Min_lr_fit, args.unfreeze_epoch)
    
    return optimizer, lr_scheduler_func, Init_lr_fit, Min_lr_fit


def create_data_loaders(args, train_lines, val_lines, clear_lines, val_clear_lines, batch_size):
    class_names, num_classes = get_classes(args.classes_path)
    
    train_dataset = YoloDataset(
        train_lines, clear_lines, args.input_shape, num_classes, 
        epoch_length=args.unfreeze_epoch, mosaic=args.mosaic, train=True
    )
    val_dataset = YoloDataset(
        val_lines, val_clear_lines, args.input_shape, num_classes,
        epoch_length=args.unfreeze_epoch, mosaic=False, train=False
    )
    
    gen = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, 
        num_workers=args.num_workers, pin_memory=True, drop_last=True, 
        collate_fn=yolo_dataset_collate
    )
    gen_val = DataLoader(
        val_dataset, shuffle=True, batch_size=batch_size, 
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=yolo_dataset_collate
    )
    
    return gen, gen_val, num_classes, class_names


def main():
    args = parse_args()
    
    if args.cuda and torch.cuda.is_available():
        torch.manual_seed(42)
        cudnn.benchmark = True
    else:
        args.cuda = False
    
    with open(args.train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(args.val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    
    clear_lines = train_lines
    val_clear_lines = val_lines
    
    num_train = len(train_lines)
    num_val = len(val_lines)
    
    class_names, num_classes = get_classes(args.classes_path)
    model = YoloBody(num_classes, args.phi)
    weights_init(model)
    
    model = load_pretrained_weights(model, args.model_path)
    
    yolo_loss = YOLOLoss(num_classes)
    loss_history = LossHistory(args.logs_dir, model, input_shape=args.input_shape)
    
    model_train = model.train()
    if args.cuda:
        model_train = torch.nn.DataParallel(model_train)
        model_train = model_train.cuda()
    
    UnFreeze_flag = False
    batch_size = args.freeze_batch_size if args.freeze_train else args.unfreeze_batch_size
    
    if args.freeze_train:
        for param in model.backbone.parameters():
            param.requires_grad = False
    
    optimizer, lr_scheduler_func, Init_lr_fit, Min_lr_fit = setup_optimizer(
        model, args, batch_size, args.init_lr, args.min_lr
    )
    
    gen, gen_val, num_classes, class_names = create_data_loaders(
        args, train_lines, val_lines, clear_lines, val_clear_lines, batch_size
    )
    
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("Dataset error!")
    
    for epoch in range(args.init_epoch, args.unfreeze_epoch):
        if epoch >= args.freeze_epoch and not UnFreeze_flag and args.freeze_train:
            batch_size = args.unfreeze_batch_size
            
            optimizer, lr_scheduler_func, Init_lr_fit, Min_lr_fit = setup_optimizer(
                model, args, batch_size, args.init_lr, args.min_lr
            )
            
            for param in model.backbone.parameters():
                param.requires_grad = True
            
            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size
            
            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("Dataset error!")
            
            gen, gen_val, _, _ = create_data_loaders(
                args, train_lines, val_lines, clear_lines, val_clear_lines, batch_size
            )
            
            UnFreeze_flag = True
        
        gen.dataset.epoch_now = epoch
        gen_val.dataset.epoch_now = epoch
        
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        
        fit_one_epoch(
            model_train, model, yolo_loss, loss_history, optimizer, epoch,
            epoch_step, epoch_step_val, gen, gen_val, args.unfreeze_epoch,
            args.cuda, args.save_period, args.logs_dir
        )


if __name__ == "__main__":
    main()
