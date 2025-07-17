"""
电化学分析模块，包含dQ/dV分析、EIS分析等功能
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit

class ElectrochemicalAnalyzer:
    """电化学分析器"""
    
    def __init__(self):
        pass
        
    def analyze_dqdv(self, 
                     voltage: np.ndarray, 
                     current: np.ndarray, 
                     time: np.ndarray,
                     smooth_window: int = 21,
                     smooth_order: int = 3) -> Dict:
        """
        分析dQ/dV曲线
        
        Args:
            voltage: 电压数据
            current: 电流数据
            time: 时间数据
            smooth_window: 平滑窗口大小
            smooth_order: 平滑多项式阶数
            
        Returns:
            Dict: 分析结果
        """
        # 计算dQ/dV
        dv = np.diff(voltage)
        dt = np.diff(time)
        dq = current[:-1] * dt
        dqdv = dq / dv
        
        # 平滑处理
        dqdv_smooth = savgol_filter(dqdv, smooth_window, smooth_order)
        
        # 峰值分析
        peaks, properties = find_peaks(dqdv_smooth, 
                                     height=np.mean(dqdv_smooth),
                                     distance=len(dqdv_smooth)//10)
        
        return {
            'dqdv': dqdv_smooth,
            'voltage': voltage[:-1],
            'peaks': peaks,
            'peak_heights': properties['peak_heights'],
            'peak_voltages': voltage[:-1][peaks]
        }
        
    def analyze_eis(self,
                    frequency: np.ndarray,
                    z_real: np.ndarray,
                    z_imag: np.ndarray) -> Dict:
        """
        分析电化学阻抗谱
        
        Args:
            frequency: 频率数据
            z_real: 实部阻抗
            z_imag: 虚部阻抗
            
        Returns:
            Dict: 分析结果
        """
        # 拟合Randles等效电路
        def randles_circuit(w, Rs, Rct, Cdl):
            Z_dl = 1 / (1j * w * Cdl)
            Z_total = Rs + (Rct * Z_dl) / (Rct + Z_dl)
            return np.hstack((Z_total.real, Z_total.imag))
            
        w = 2 * np.pi * frequency
        Z = np.hstack((z_real, z_imag))
        
        # 初始参数估计
        p0 = [np.min(z_real), np.max(z_real)-np.min(z_real), 1e-6]
        
        # 拟合
        popt, _ = curve_fit(randles_circuit, w, Z, p0=p0)
        Rs, Rct, Cdl = popt
        
        return {
            'Rs': Rs,  # 溶液电阻
            'Rct': Rct,  # 电荷转移电阻
            'Cdl': Cdl,  # 双层电容
            'Z_fit': randles_circuit(w, Rs, Rct, Cdl)
        }
        
    def analyze_cv(self,
                   voltage: np.ndarray,
                   current: np.ndarray,
                   scan_rate: float) -> Dict:
        """
        分析循环伏安曲线
        
        Args:
            voltage: 电压数据
            current: 电流数据
            scan_rate: 扫描速率 (V/s)
            
        Returns:
            Dict: 分析结果
        """
        # 寻找氧化还原峰
        peaks_ox, _ = find_peaks(current, height=np.mean(current))
        peaks_red, _ = find_peaks(-current, height=np.mean(-current))
        
        # 计算峰电位差
        if len(peaks_ox) > 0 and len(peaks_red) > 0:
            delta_ep = voltage[peaks_ox[0]] - voltage[peaks_red[0]]
        else:
            delta_ep = None
            
        # 计算峰电流比
        if len(peaks_ox) > 0 and len(peaks_red) > 0:
            peak_ratio = abs(current[peaks_ox[0]] / current[peaks_red[0]])
        else:
            peak_ratio = None
            
        return {
            'oxidation_peaks': peaks_ox,
            'reduction_peaks': peaks_red,
            'peak_potential_diff': delta_ep,
            'peak_current_ratio': peak_ratio,
            'scan_rate': scan_rate
        } 