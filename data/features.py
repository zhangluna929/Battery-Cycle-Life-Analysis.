"""
特征工程模块，用于提取电池数据的关键特征
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis

class FeatureExtractor:
    """电池数据特征提取器"""
    
    def __init__(self):
        self.feature_functions = {
            'statistical': self._extract_statistical_features,
            'cycling': self._extract_cycling_features,
            'electrochemical': self._extract_electrochemical_features
        }
        
    def extract_features(self, 
                        data: pd.DataFrame, 
                        feature_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        提取特征
        
        Args:
            data: 输入数据表
            feature_types: 要提取的特征类型列表，默认提取所有特征
            
        Returns:
            DataFrame: 提取的特征
        """
        if feature_types is None:
            feature_types = list(self.feature_functions.keys())
            
        features = {}
        for feat_type in feature_types:
            if feat_type not in self.feature_functions:
                raise ValueError(f"不支持的特征类型: {feat_type}")
            features.update(self.feature_functions[feat_type](data))
            
        return pd.DataFrame(features, index=[0])
        
    def _extract_statistical_features(self, data: pd.DataFrame) -> Dict:
        """提取统计特征"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        features = {}
        
        for col in numeric_cols:
            col_data = data[col].values
            features.update({
                f'{col}_mean': np.mean(col_data),
                f'{col}_std': np.std(col_data),
                f'{col}_skew': skew(col_data),
                f'{col}_kurtosis': kurtosis(col_data)
            })
            
        return features
        
    def _extract_cycling_features(self, data: pd.DataFrame) -> Dict:
        """提取循环特征"""
        features = {}
        
        if 'capacity' in data.columns:
            capacity = data['capacity'].values
            features.update({
                'initial_capacity': capacity[0],
                'capacity_retention': capacity[-1] / capacity[0],
                'capacity_fade_rate': (capacity[0] - capacity[-1]) / len(capacity)
            })
            
        if 'voltage' in data.columns and 'current' in data.columns:
            features['coulombic_efficiency'] = np.abs(
                data['current'][data['current'] < 0].sum() / 
                data['current'][data['current'] > 0].sum()
            )
            
        return features
        
    def _extract_electrochemical_features(self, data: pd.DataFrame) -> Dict:
        """提取电化学特征"""
        features = {}
        
        if 'voltage' in data.columns and 'current' in data.columns:
            # 计算dQ/dV
            voltage = data['voltage'].values
            current = data['current'].values
            dv = np.diff(voltage)
            dq = np.diff(current * np.diff(data.index.values))
            dqdv = dq / dv
            
            # 寻找特征峰
            peaks, _ = find_peaks(dqdv, height=np.mean(dqdv))
            if len(peaks) > 0:
                features.update({
                    'peak_count': len(peaks),
                    'max_peak_height': np.max(dqdv[peaks]),
                    'mean_peak_height': np.mean(dqdv[peaks])
                })
                
        return features 