"""
基础绘图模块
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
from typing import Optional, List, Dict, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Plotter:
    """绘图工具类"""
    
    def __init__(self, style: str = 'seaborn'):
        """
        初始化绘图器
        
        Args:
            style: matplotlib样式
        """
        plt.style.use(style)
        
    def plot_capacity_fade(self,
                          cycles: np.ndarray,
                          capacity: np.ndarray,
                          ax: Optional[Axes] = None,
                          label: Optional[str] = None,
                          show_fit: bool = True) -> Axes:
        """
        绘制容量衰减曲线
        
        Args:
            cycles: 循环数
            capacity: 容量数据
            ax: matplotlib轴对象
            label: 曲线标签
            show_fit: 是否显示拟合曲线
            
        Returns:
            matplotlib轴对象
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
            
        # 绘制实际数据
        ax.scatter(cycles, capacity, s=20, alpha=0.5, label=f'{label} (实测)' if label else '实测')
        
        if show_fit:
            # 指数衰减拟合
            from scipy.optimize import curve_fit
            
            def exp_decay(x, a, b, c):
                return a * np.exp(-b * x) + c
                
            popt, _ = curve_fit(exp_decay, cycles, capacity)
            cycles_fit = np.linspace(cycles.min(), cycles.max(), 100)
            capacity_fit = exp_decay(cycles_fit, *popt)
            
            ax.plot(cycles_fit, capacity_fit, '--', 
                   label=f'{label} (拟合)' if label else '拟合')
            
        ax.set_xlabel('循环数')
        ax.set_ylabel('容量 (Ah)')
        ax.grid(True, alpha=0.3)
        if label or show_fit:
            ax.legend()
            
        return ax
        
    def plot_dqdv(self,
                  voltage: np.ndarray,
                  dqdv: np.ndarray,
                  peaks: Optional[np.ndarray] = None,
                  ax: Optional[Axes] = None) -> Axes:
        """
        绘制dQ/dV曲线
        
        Args:
            voltage: 电压数据
            dqdv: dQ/dV数据
            peaks: 峰值位置
            ax: matplotlib轴对象
            
        Returns:
            matplotlib轴对象
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
            
        ax.plot(voltage, dqdv, 'b-', label='dQ/dV')
        
        if peaks is not None:
            ax.plot(voltage[peaks], dqdv[peaks], 'ro', label='峰值')
            
        ax.set_xlabel('电压 (V)')
        ax.set_ylabel('dQ/dV (Ah/V)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax
        
    def plot_eis(self,
                 z_real: np.ndarray,
                 z_imag: np.ndarray,
                 fit_result: Optional[Dict] = None,
                 ax: Optional[Axes] = None) -> Axes:
        """
        绘制阻抗谱
        
        Args:
            z_real: 实部阻抗
            z_imag: 虚部阻抗
            fit_result: 拟合结果
            ax: matplotlib轴对象
            
        Returns:
            matplotlib轴对象
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))
            
        ax.plot(z_real, -z_imag, 'bo-', label='测量值')
        
        if fit_result is not None:
            z_fit = fit_result['Z_fit']
            n_points = len(z_real)
            ax.plot(z_fit[:n_points], -z_fit[n_points:], 'r--', label='拟合')
            
            # 添加拟合参数
            text = f"Rs = {fit_result['Rs']:.2f} Ω\n"
            text += f"Rct = {fit_result['Rct']:.2f} Ω\n"
            text += f"Cdl = {fit_result['Cdl']*1e6:.2f} µF"
            ax.text(0.95, 0.05, text,
                   transform=ax.transAxes,
                   verticalalignment='bottom',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
        ax.set_xlabel("Z' (Ω)")
        ax.set_ylabel("-Z'' (Ω)")
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend()
        
        return ax
        
    def plot_cv(self,
                voltage: np.ndarray,
                current: np.ndarray,
                scan_rate: float,
                peaks: Optional[Dict] = None,
                ax: Optional[Axes] = None) -> Axes:
        """
        绘制循环伏安曲线
        
        Args:
            voltage: 电压数据
            current: 电流数据
            scan_rate: 扫描速率
            peaks: 峰值信息
            ax: matplotlib轴对象
            
        Returns:
            matplotlib轴对象
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
            
        ax.plot(voltage, current, 'b-', label=f'{scan_rate} mV/s')
        
        if peaks is not None:
            if 'oxidation_peaks' in peaks:
                ax.plot(voltage[peaks['oxidation_peaks']],
                       current[peaks['oxidation_peaks']],
                       'ro', label='氧化峰')
            if 'reduction_peaks' in peaks:
                ax.plot(voltage[peaks['reduction_peaks']],
                       current[peaks['reduction_peaks']],
                       'go', label='还原峰')
                       
            # 添加峰电位差和峰电流比
            if peaks.get('peak_potential_diff') is not None:
                text = f"ΔEp = {peaks['peak_potential_diff']*1000:.1f} mV\n"
                if peaks.get('peak_current_ratio') is not None:
                    text += f"Ipa/Ipc = {peaks['peak_current_ratio']:.2f}"
                ax.text(0.95, 0.05, text,
                       transform=ax.transAxes,
                       verticalalignment='bottom',
                       horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
        ax.set_xlabel('电压 (V)')
        ax.set_ylabel('电流 (A)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax
        
    def create_interactive_plot(self,
                              data: Dict[str, np.ndarray],
                              plot_type: str = 'scatter',
                              title: str = '') -> go.Figure:
        """
        创建交互式图表
        
        Args:
            data: 绘图数据
            plot_type: 图表类型
            title: 图表标题
            
        Returns:
            plotly图表对象
        """
        fig = go.Figure()
        
        if plot_type == 'scatter':
            for name, values in data.items():
                if len(values.shape) == 1:
                    fig.add_trace(go.Scatter(
                        y=values,
                        name=name,
                        mode='lines+markers'
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=values[:,0],
                        y=values[:,1],
                        name=name,
                        mode='lines+markers'
                    ))
                    
        elif plot_type == 'heatmap':
            fig = go.Figure(data=go.Heatmap(z=data))
            
        fig.update_layout(
            title=title,
            xaxis_title='样本',
            yaxis_title='数值',
            hovermode='x'
        )
        
        return fig 