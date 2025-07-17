"""
模型模块测试
"""

import pytest
import torch
import numpy as np
from battery_cycle_life.models import BatteryNet, Trainer, Predictor

@pytest.fixture
def sample_data():
    """生成测试数据"""
    np.random.seed(42)
    n_samples = 100
    seq_len = 20
    input_dim = 10
    
    # 生成序列数据
    X = np.random.randn(n_samples, seq_len, input_dim)
    y = np.sum(X[:, -5:, :], axis=(1,2))  # 使用最后5个时间步的和作为目标
    y = y.reshape(-1, 1)
    
    return torch.FloatTensor(X), torch.FloatTensor(y)

@pytest.fixture
def model():
    """创建测试模型"""
    return BatteryNet(
        input_dim=10,
        hidden_dim=32,
        num_layers=2,
        dropout=0.1
    )

def test_battery_net(model, sample_data):
    """测试电池网络"""
    X, _ = sample_data
    batch_size = 16
    
    # 测试前向传播
    output, attention = model(X[:batch_size])
    assert output.shape == (batch_size, 1)
    assert attention.shape[1] == attention.shape[2] == X.shape[1]  # 注意力矩阵形状
    
    # 测试带掩码的前向传播
    mask = torch.ones(batch_size, X.shape[1], dtype=torch.bool)
    mask[:, -5:] = False  # 掩盖最后5个时间步
    output, attention = model(X[:batch_size], mask)
    assert output.shape == (batch_size, 1)
    
def test_trainer(model, sample_data):
    """测试训练器"""
    X, y = sample_data
    trainer = Trainer(model)
    
    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16
    )
    
    # 测试训练
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=2,
        early_stopping=None
    )
    
    assert 'train_loss' in history
    assert 'val_loss' in history
    assert len(history['train_loss']) == 2
    assert len(history['val_loss']) == 2
    
    # 测试模型保存和加载
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, 'model.pt')
        trainer.save_model(model_path)
        assert os.path.exists(model_path)
        
        trainer.load_model(model_path)
        
def test_predictor(model, sample_data):
    """测试预测器"""
    X, _ = sample_data
    predictor = Predictor(model)
    
    # 测试基本预测
    predictions = predictor.predict(X)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(X), 1)
    
    # 测试带不确定度的预测
    predictions, uncertainties = predictor.predict(
        X,
        return_uncertainty=True,
        n_samples=10
    )
    assert isinstance(predictions, np.ndarray)
    assert isinstance(uncertainties, np.ndarray)
    assert predictions.shape == uncertainties.shape
    
    # 测试带解释的预测
    result = predictor.predict_with_explanation(X[:1])
    assert 'prediction' in result
    assert 'attention_weights' in result
    
    # 测试特征重要性
    result = predictor.get_feature_importance(
        X[:1],
        feature_names=[f'feature_{i}' for i in range(X.shape[-1])]
    )
    assert 'importance_scores' in result
    assert 'feature_importance' in result
    assert len(result['importance_scores']) == X.shape[-1] 