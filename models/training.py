"""
模型训练模块
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Callable
import numpy as np
from tqdm import tqdm

class Trainer:
    """模型训练器"""
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optional[optim.Optimizer] = None,
                 criterion: Optional[nn.Module] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化训练器
        
        Args:
            model: 待训练的模型
            optimizer: 优化器
            criterion: 损失函数
            device: 训练设备
        """
        self.model = model.to(device)
        self.device = device
        
        # 默认优化器
        if optimizer is None:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=1e-3,
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer
            
        # 默认损失函数
        if criterion is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion
            
        # 训练状态记录
        self.train_losses = []
        self.val_losses = []
        self.best_loss = float('inf')
        self.best_model = None
        
    def train_epoch(self,
                    train_loader: DataLoader,
                    epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch}') as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                
                if isinstance(output, tuple):
                    output = output[0]  # 取预测值
                    
                loss = self.criterion(output, target)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
        return total_loss / len(train_loader)
        
    def validate(self, val_loader: DataLoader) -> float:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                if isinstance(output, tuple):
                    output = output[0]
                    
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
        
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int,
              scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
              early_stopping: Optional[int] = None,
              save_best: bool = True) -> Dict:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            scheduler: 学习率调度器
            early_stopping: 早停轮数
            save_best: 是否保存最佳模型
            
        Returns:
            训练历史
        """
        no_improve = 0
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # 更新学习率
            if scheduler is not None:
                scheduler.step()
                
            # 保存最佳模型
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                if save_best:
                    self.best_model = self.model.state_dict()
                no_improve = 0
            else:
                no_improve += 1
                
            # 早停
            if early_stopping is not None and no_improve >= early_stopping:
                print(f'Early stopping at epoch {epoch}')
                break
                
            print(f'Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')
            
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'best_loss': self.best_loss
        }
        
    def save_model(self, path: str):
        """保存模型"""
        if self.best_model is not None:
            torch.save(self.best_model, path)
        else:
            torch.save(self.model.state_dict(), path)
            
    def load_model(self, path: str):
        """加载模型"""
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device) 