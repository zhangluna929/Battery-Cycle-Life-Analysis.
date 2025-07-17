"""
分析模块测试
"""

import pytest
import numpy as np
from battery_cycle_life.analysis import (
    ElectrochemicalAnalyzer,
    KineticsAnalyzer,
    StructuralAnalyzer
)

@pytest.fixture
def sample_data():
    """生成测试数据"""
    np.random.seed(42)
    n_samples = 1000
    
    # 生成电化学测试数据
    voltage = np.linspace(2.5, 4.2, n_samples)
    current = np.sin(2*np.pi*voltage) + np.random.normal(0, 0.1, n_samples)
    time = np.arange(n_samples)
    
    # 生成EIS测试数据
    frequency = np.logspace(0, 5, 100)
    z_real = 0.1 + 0.5/(1 + (2*np.pi*frequency*1e-6)**2)
    z_imag = -0.5*2*np.pi*frequency*1e-6/(1 + (2*np.pi*frequency*1e-6)**2)
    z_imag += np.random.normal(0, 0.01, 100)
    
    return {
        'voltage': voltage,
        'current': current,
        'time': time,
        'frequency': frequency,
        'z_real': z_real,
        'z_imag': z_imag
    }

def test_electrochemical_analyzer(sample_data):
    """测试电化学分析器"""
    analyzer = ElectrochemicalAnalyzer()
    
    # 测试dQ/dV分析
    result = analyzer.analyze_dqdv(
        sample_data['voltage'],
        sample_data['current'],
        sample_data['time']
    )
    assert 'dqdv' in result
    assert 'peaks' in result
    assert isinstance(result['dqdv'], np.ndarray)
    
    # 测试EIS分析
    result = analyzer.analyze_eis(
        sample_data['frequency'],
        sample_data['z_real'],
        sample_data['z_imag']
    )
    assert 'Rs' in result
    assert 'Rct' in result
    assert 'Cdl' in result
    assert isinstance(result['Z_fit'], np.ndarray)
    
def test_kinetics_analyzer(sample_data):
    """测试动力学分析器"""
    analyzer = KineticsAnalyzer()
    
    # 测试扩散系数计算
    result = analyzer.calculate_diffusion_coefficient(
        sample_data['time'][:100],
        sample_data['voltage'][:100],
        sample_data['current'][:100]
    )
    assert 'diffusion_coefficient' in result
    assert result['diffusion_coefficient'] > 0
    
    # 测试反应动力学分析
    temperature = np.array([298, 308, 318, 328])
    rate_constant = 0.001 * np.exp(-50000/(8.314*temperature))
    result = analyzer.analyze_reaction_kinetics(temperature, rate_constant)
    assert 'activation_energy' in result
    assert 'pre_exponential_factor' in result
    assert result['activation_energy'] > 0
    
def test_structural_analyzer():
    """测试结构分析器"""
    analyzer = StructuralAnalyzer()
    
    # 生成XRD测试数据
    two_theta = np.linspace(10, 80, 1000)
    intensity = (
        100 * np.exp(-(two_theta - 30)**2/2) +
        80 * np.exp(-(two_theta - 45)**2/2) +
        60 * np.exp(-(two_theta - 60)**2/2)
    )
    intensity += np.random.normal(0, 5, 1000)
    
    # 测试XRD分析
    result = analyzer.analyze_xrd(two_theta, intensity)
    assert 'peak_positions' in result
    assert 'peak_intensities' in result
    assert 'd_spacing' in result
    assert len(result['peak_positions']) >= 3  # 应该检测到至少3个峰
    
    # 测试晶体结构分析
    result = analyzer.analyze_crystal_structure(
        lattice_constant=3.615,  # Li金属的晶格常数
        atomic_positions=[(0,0,0), (0.5,0.5,0.5)]  # BCC结构
    )
    assert 'volume' in result
    assert 'density' in result
    assert result['n_atoms'] == 2 