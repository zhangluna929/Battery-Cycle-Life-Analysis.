"""
数据处理模块测试
"""

import pytest
import numpy as np
import pandas as pd
from battery_cycle_life.data import DataLoader, Preprocessor, FeatureExtractor

@pytest.fixture
def sample_data():
    """生成测试数据"""
    np.random.seed(42)
    n_samples = 100
    
    data = pd.DataFrame({
        'cycle': np.arange(n_samples),
        'voltage': 3.5 + np.random.normal(0, 0.1, n_samples),
        'current': np.random.normal(0, 1, n_samples),
        'capacity': 2.5 * np.exp(-0.001 * np.arange(n_samples)) + np.random.normal(0, 0.05, n_samples)
    })
    
    return data

def test_data_loader():
    """测试数据加载器"""
    loader = DataLoader()
    
    # 测试支持的格式
    assert 'csv' in loader.supported_formats
    assert 'xlsx' in loader.supported_formats
    
    # 测试格式验证
    with pytest.raises(ValueError):
        loader.load('test.unknown')
        
def test_preprocessor(sample_data):
    """测试数据预处理器"""
    preprocessor = Preprocessor()
    
    # 测试预处理
    processed_data = preprocessor.fit_transform(sample_data)
    assert isinstance(processed_data, pd.DataFrame)
    assert processed_data.shape[0] <= sample_data.shape[0]  # 可能去除了异常值
    
    # 测试transform
    new_data = preprocessor.transform(sample_data)
    assert isinstance(new_data, pd.DataFrame)
    
def test_feature_extractor(sample_data):
    """测试特征提取器"""
    extractor = FeatureExtractor()
    
    # 测试特征提取
    features = extractor.extract_features(sample_data)
    assert isinstance(features, pd.DataFrame)
    assert features.shape[0] == 1  # 一个样本的特征
    
    # 测试指定特征类型
    features = extractor.extract_features(
        sample_data,
        feature_types=['statistical']
    )
    assert isinstance(features, pd.DataFrame)
    
    # 测试无效特征类型
    with pytest.raises(ValueError):
        extractor.extract_features(
            sample_data,
            feature_types=['invalid_type']
        ) 