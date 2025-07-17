"""
模型推理模块
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from torch.utils.data import DataLoader, TensorDataset

class Predictor:
    """模型预测器"""
    
    def __init__(self,
                 model: torch.nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化预测器
        
        Args:
            model: 训练好的模型
            device: 推理设备
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
    def predict(self,
                data: Union[torch.Tensor, np.ndarray],
                batch_size: int = 32,
                return_uncertainty: bool = False,
                n_samples: int = 100) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        模型预测
        
        Args:
            data: 输入数据
            batch_size: 批次大小
            return_uncertainty: 是否返回不确定度
            n_samples: MC Dropout采样次数
            
        Returns:
            预测结果，如果return_uncertainty为True则同时返回不确定度
        """
        # 数据转换
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
            
        # 创建数据加载器
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions: list[np.ndarray] = []
        uncertainties_list: list[np.ndarray] = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch[0].to(self.device)
                
                if return_uncertainty:
                    # MC Dropout采样
                    self.model.train()  # 启用dropout
                    samples = []
                    for _ in range(n_samples):
                        output = self.model(batch)
                        if isinstance(output, tuple):
                            output = output[0]
                        samples.append(output)
                    samples = torch.stack(samples)  # [n_samples, batch_size, 1]
                    
                    # 计算均值和标准差
                    mean = samples.mean(dim=0)
                    std = samples.std(dim=0)
                    
                    predictions.append(mean.cpu().numpy())
                    uncertainties_list.append(std.cpu().numpy())
                    
                else:
                    self.model.eval()
                    output = self.model(batch)
                    if isinstance(output, tuple):
                        output = output[0]
                    predictions.append(output.cpu().numpy())
                    
        preds = np.concatenate(predictions) if predictions else np.array([])
        if return_uncertainty:
            unc = np.concatenate(uncertainties_list) if uncertainties_list else np.array([])
            return preds, unc
        return preds
        
    def predict_with_explanation(self,
                               data: Union[torch.Tensor, np.ndarray],
                               return_attention: bool = True) -> Dict:
        """
        带解释的预测
        
        Args:
            data: 输入数据
            return_attention: 是否返回注意力权重
            
        Returns:
            预测结果和解释信息
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        data = data.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(data)
            
        result = {'prediction': output[0].cpu().numpy()}
        
        if return_attention and len(output) > 1:
            result['attention_weights'] = output[1].cpu().numpy()
            
        return result
        
    @torch.no_grad()
    def get_feature_importance(self,
                             data: Union[torch.Tensor, np.ndarray],
                             feature_names: Optional[List[str]] = None) -> Dict:
        """
        计算特征重要性
        
        Args:
            data: 输入数据
            feature_names: 特征名列表
            
        Returns:
            特征重要性分析结果
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        data = data.to(self.device)
        
        # 基准预测
        base_pred = self.model(data)[0]
        
        # 特征重要性计算
        importance = []
        for i in range(data.shape[-1]):
            # 特征置零
            perturbed_data = data.clone()
            perturbed_data[..., i] = 0
            
            # 计算预测变化
            pred = self.model(perturbed_data)[0]
            importance.append(torch.abs(base_pred - pred).mean().item())
            
        importance = np.array(importance)
        
        # 归一化
        importance = importance / importance.sum()
        
        result = {
            'importance_scores': importance
        }
        
        if feature_names is not None:
            result['feature_importance'] = dict(zip(feature_names, importance))
            
        return result 