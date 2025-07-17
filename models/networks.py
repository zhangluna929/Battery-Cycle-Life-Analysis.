"""
神经网络架构模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class BatteryNet(nn.Module):
    """电池寿命预测网络"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        初始化网络
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐层维度
            num_layers: LSTM层数
            dropout: Dropout比率
        """
        super().__init__()
        
        # 特征提取
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout
        )
        
        # 输出层
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1)
        )
        
    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入数据 [batch_size, seq_len, input_dim]
            mask: 掩码 [batch_size, seq_len]
            
        Returns:
            预测结果和注意力权重
        """
        # 特征提取
        batch_size, seq_len, _ = x.shape
        x = self.feature_extractor(x)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        
        # 自注意力
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len]
        attn_out, attn_weights = self.attention(
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
            key_padding_mask=mask
        )
        attn_out = attn_out.transpose(0, 1)
        
        # 取序列最后一个时间步
        out = self.regressor(attn_out[:, -1, :])
        
        return out, attn_weights
        
class UncertaintyNet(BatteryNet):
    """具有不确定度估计的电池寿命预测网络"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = kwargs.get('hidden_dim', 128)
        
        # 方差预测器
        self.variance_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(kwargs.get('dropout', 0.1)),
            nn.Linear(hidden_dim//2, 1),
            nn.Softplus()  # 确保方差为正
        )
        
    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            预测均值、方差和注意力权重
        """
        mean, attn_weights = super().forward(x, mask)
        variance = self.variance_regressor(x[:, -1, :])
        return mean, variance, attn_weights 