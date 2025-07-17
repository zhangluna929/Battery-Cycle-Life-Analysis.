"""
数据预处理模块，包含数据清洗、标准化等功能
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest

class Preprocessor:
    """数据预处理器，提供数据清洗和标准化功能"""
    
    def __init__(self):
        self.scaler = None
        self.outlier_detector = None
        
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        拟合并转换数据
        
        Args:
            data: 输入数据表
            
        Returns:
            DataFrame: 预处理后的数据
        """
        # 1. 清洗数据
        data = self._clean_data(data)
        
        # 2. 检测异常值
        data = self._handle_outliers(data)
        
        # 3. 标准化
        data = self._standardize(data)
        
        return data
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        使用已拟合的参数转换数据
        
        Args:
            data: 输入数据表
            
        Returns:
            DataFrame: 预处理后的数据
        """
        if self.scaler is None:
            raise ValueError("请先调用fit_transform")
            
        data = self._clean_data(data)
        if self.outlier_detector is not None:
            mask = self.outlier_detector.predict(data) == 1
            data = data[mask]
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = self.scaler.transform(data[numeric_cols])
        
        return data
        
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """基础数据清洗"""
        # 1. 删除重复行
        data = data.drop_duplicates()
        
        # 2. 处理缺失值
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(method='ffill')
        
        return data
        
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """异常值检测与处理"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        self.outlier_detector = IsolationForest(contamination=0.1, random_state=42)
        mask = self.outlier_detector.fit_predict(data[numeric_cols]) == 1
        return data[mask]
        
    def _standardize(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据标准化"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        self.scaler = RobustScaler()
        data[numeric_cols] = self.scaler.fit_transform(data[numeric_cols])
        return data 