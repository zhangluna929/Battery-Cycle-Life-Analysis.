"""
数据加载模块，支持多种数据格式的导入和处理
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Optional

class DataLoader:
    """数据加载器，支持多种电池测试仪器数据格式"""
    
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'txt', 'mat']
        
    def load(self, file_path: str, format: Optional[str] = None) -> pd.DataFrame:
        """
        加载数据文件
        
        Args:
            file_path: 数据文件路径
            format: 文件格式，如果为None则自动识别
            
        Returns:
            DataFrame: 标准化的数据表
        """
        if format is None:
            format = file_path.split('.')[-1].lower()
            
        if format not in self.supported_formats:
            raise ValueError(f"不支持的文件格式: {format}")
            
        if format == 'csv':
            return self._load_csv(file_path)
        elif format == 'xlsx':
            return self._load_excel(file_path)
        elif format == 'txt':
            return self._load_text(file_path)
        elif format == 'mat':
            return self._load_matlab(file_path)
            
    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """加载CSV文件"""
        return pd.read_csv(file_path)
        
    def _load_excel(self, file_path: str) -> pd.DataFrame:
        """加载Excel文件"""
        return pd.read_excel(file_path)
        
    def _load_text(self, file_path: str) -> pd.DataFrame:
        """加载文本文件"""
        return pd.read_csv(file_path, delimiter='\t')
        
    def _load_matlab(self, file_path: str) -> pd.DataFrame:
        """加载Matlab文件"""
        from scipy.io import loadmat
        data = loadmat(file_path)
        # 处理matlab数据结构
        return pd.DataFrame(data) 