import torch
from tqdm import tqdm
import torch.nn as nn
from typing import Tuple, List, Union
from utils.utils import get_lr

def fit_one_epoch(
    model_train: nn.Module,
    model: nn.Module,
    loss_history: object,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    epoch_step: int,
    epoch_step_val: int,
    gen: iter,
    gen_val: iter,
    Epoch: int,
    cuda: bool,
    save_period: int,
    logs_dir: str
) -> None:
    """训练一个epoch的简化版本"""
    
    # 训练阶段
    model_train.train()
    train_loss = 0.0
    criterion = nn.MSELoss()
    
    print('Start Training')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', 
             postfix=dict, mininterval=0.3) as pbar:
        
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
                
            # 准备数据
            images = batch[0]
            images = torch.from_numpy(images).type(torch.FloatTensor)
            
            if cuda:
                images = images.cuda()
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model_train(images)
            loss_value = criterion(outputs,images)
            
            # 反向传播
            loss_value.backward()
            optimizer.step()
            
            # 记录损失
            train_loss += loss_value.item()
            avg_loss = train_loss / (iteration + 1)
            pbar.set_postfix(**{'loss': avg_loss})
            pbar.update(1)
    
    # 验证阶段
    model_train.eval()
    val_loss = 0.0
    
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', 
             postfix=dict, mininterval=0.3) as pbar:
        
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
                
            # 准备数据
            images = batch[0]
            images = torch.from_numpy(images).type(torch.FloatTensor)
            
            if cuda:
                images = images.cuda()
            
            # 前向传播（无梯度）
            with torch.no_grad():
                outputs = model_train(images)
                loss_value = criterion(outputs, images)
            
            # 记录损失
            val_loss += loss_value.item()
            avg_loss = val_loss / (iteration + 1)
            pbar.set_postfix(**{'val_loss': avg_loss})
            pbar.update(1)
    
    # 计算平均损失
    avg_train_loss = train_loss / epoch_step
    avg_val_loss = val_loss / epoch_step_val
    
    # 记录和保存
    loss_history.append_loss(epoch + 1, avg_train_loss, avg_val_loss)
    print(f'Epoch: {epoch + 1}/{Epoch}')
    print(f'Total Loss: {avg_train_loss:.3f} || Val Loss: {avg_val_loss:.3f}')
    
    # 保存模型
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        filename = f'{logs_dir}ep{epoch+1:03d}-loss{avg_train_loss:.3f}-val_loss{avg_val_loss:.3f}.pth'
        torch.save(model.state_dict(), filename)
        print(f'Model saved: {filename}')