import datetime
import os
from pathlib import Path
from typing import Optional

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class LossHistory:
    """记录和可视化训练损失的类"""
    
    def __init__(self, log_dir: str, model: torch.nn.Module, input_shape: tuple):
        """初始化损失历史记录器
        
        Args:
            log_dir: 日志目录路径
            model: 要记录计算图的模型
            input_shape: 输入张量形状 (height, width)
        """
        time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.log_dir = Path(log_dir) / f"loss_{time_str}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化损失记录
        self.losses = []
        self.val_loss = []
        
        # 初始化TensorBoard写入器
        self.writer = SummaryWriter(self.log_dir)
        self._add_model_graph(model, input_shape)
    
    def _add_model_graph(self, model: torch.nn.Module, input_shape: tuple) -> None:
        """添加模型计算图到TensorBoard"""
        try:
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except Exception as e:
            print(f"Failed to add model graph to TensorBoard: {e}")
    
    def append_loss(self, epoch: int, loss: float, val_loss: float) -> None:
        """记录每个epoch的损失值
        
        Args:
            epoch: 当前epoch数
            loss: 训练损失
            val_loss: 验证损失
        """
        # 记录损失值
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        
        # 保存损失到文件
        self._save_loss_to_file("epoch_loss.txt", loss)
        self._save_loss_to_file("epoch_val_loss.txt", val_loss)
        
        # 写入TensorBoard
        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        
        # 绘制损失曲线
        self.plot_loss()
    
    def _save_loss_to_file(self, filename: str, loss_value: float) -> None:
        """将损失值保存到文本文件"""
        file_path = self.log_dir / filename
        with open(file_path, 'a') as f:
            f.write(f"{loss_value}\n")
    
    def plot_loss(self) -> None:
        """绘制并保存损失曲线图"""
        if len(self.losses) == 0:
            return
            
        iters = range(len(self.losses))
        
        plt.figure(figsize=(10, 6))
        
        # 绘制原始损失曲线
        plt.plot(iters, self.losses, 'red', linewidth=2, label='Train Loss')
        plt.plot(iters, self.val_loss, 'blue', linewidth=2, label='Val Loss')
        
        # 尝试绘制平滑后的损失曲线
        self._plot_smoothed_losses(iters)
        
        # 设置图表属性
        plt.grid(True, alpha=0.3)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(self.log_dir / "epoch_loss.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_smoothed_losses(self, iters: range) -> None:
        """绘制平滑后的损失曲线"""
        try:
            # 根据数据点数量确定平滑窗口大小
            if len(self.losses) < 5:  # 数据点太少时不进行平滑
                return
                
            window_size = 5 if len(self.losses) < 25 else 15
            
            if len(self.losses) > window_size:
                smoothed_train_loss = scipy.signal.savgol_filter(self.losses, window_size, 3)
                smoothed_val_loss = scipy.signal.savgol_filter(self.val_loss, window_size, 3)
                
                plt.plot(iters, smoothed_train_loss, 'green', linestyle='--', 
                        linewidth=2, label='Smoothed Train Loss', alpha=0.7)
                plt.plot(iters, smoothed_val_loss, 'orange', linestyle='--', 
                        linewidth=2, label='Smoothed Val Loss', alpha=0.7)
        except Exception as e:
            print(f"Smoothing failed: {e}")
    
    def close(self) -> None:
        """关闭TensorBoard写入器"""
        self.writer.close()
    
    def get_latest_losses(self, n: int = 5) -> tuple:
        """获取最近n个损失值
        
        Args:
            n: 要获取的损失值数量
            
        Returns:
            tuple: (最近n个训练损失, 最近n个验证损失)
        """
        n = min(n, len(self.losses))
        return self.losses[-n:], self.val_loss[-n:]
