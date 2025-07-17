"""
动力学分析模块
提供电池扩散系数计算（GITT/PITT）和反应动力学（Arrhenius）分析
"""

from __future__ import annotations

import numpy as np
from typing import Dict
from scipy.optimize import curve_fit

__all__ = ["KineticsAnalyzer"]

class KineticsAnalyzer:
    """电池动力学分析工具类"""

    def __init__(self) -> None:
        pass

    # ---------------------------------------------------------------------
    # Diffusion coefficient estimation
    # ---------------------------------------------------------------------
    def calculate_diffusion_coefficient(
        self,
        time: np.ndarray,
        voltage: np.ndarray,
        current: np.ndarray,
        method: str = "GITT",
    ) -> Dict:
        """根据电压/电流响应计算固相扩散系数。

        参数
        ------
        time: np.ndarray
            时间序列 (s)
        voltage: np.ndarray
            电压序列 (V)
        current: np.ndarray
            电流序列 (A)
        method: str, default ``"GITT"``
            使用的测试方法，可选 ``"GITT"`` 或 ``"PITT"``
        """
        method = method.upper()
        if method == "GITT":
            return self._gitt(time, voltage, current)
        if method == "PITT":
            return self._pitt(time, voltage, current)
        raise ValueError("method must be 'GITT' or 'PITT'")

    # ------------------------------- private -----------------------------
    def _gitt(self, t: np.ndarray, E: np.ndarray, I: np.ndarray) -> Dict:
        """基于 GITT 的扩散系数估算 (简化形式)。"""
        # 稳态与瞬态电压差
        delta_es = E[-1] - E[0]
        delta_et = np.max(E) - np.min(E)
        # 时间常数
        tau = t[-1] - t[0]
        # 假设扩散长度 L = 100 µm (1e-4 m)
        L = 1e-4
        D = 4 / np.pi * (L**2 / tau) * (delta_es / delta_et) ** 2
        return {
            "method": "GITT",
            "diffusion_coefficient": D,
            "tau": tau,
            "delta_es": delta_es,
            "delta_et": delta_et,
        }

    def _pitt(self, t: np.ndarray, E: np.ndarray, I: np.ndarray) -> Dict:
        """基于 PITT 的扩散系数估算 (指数衰减拟合)。"""
        # 忽略前 5 个瞬态点以避免激励尖峰
        t_fit = t[5:] - t[5]
        i_fit = I[5:]

        def current_decay(t, i0, D):
            return i0 * np.exp(-D * t)

        p0 = [i_fit[0], 1e-6]
        popt, _ = curve_fit(current_decay, t_fit, i_fit, p0=p0, maxfev=10000)
        i0, D = popt
        return {
            "method": "PITT",
            "diffusion_coefficient": D,
            "i0": i0,
        }

    # ---------------------------------------------------------------------
    # Reaction kinetics (Arrhenius)
    # ---------------------------------------------------------------------
    def analyze_reaction_kinetics(
        self, temperature: np.ndarray, rate_constant: np.ndarray
    ) -> Dict:
        """Arrhenius 拟合以获得活化能 Ea 与指前因子 A。"""
        if temperature.shape != rate_constant.shape:
            raise ValueError("temperature and rate_constant must have same shape")
        T_inv = 1.0 / temperature
        ln_k = np.log(rate_constant)
        # 线性拟合 ln k = ln A - Ea/R * 1/T
        slope, intercept = np.polyfit(T_inv, ln_k, 1)
        R = 8.314  # J/mol/K
        Ea = -slope * R
        A = np.exp(intercept)
        # 计算拟合优度 R²
        corr = np.corrcoef(T_inv, ln_k)[0, 1]
        r2 = corr**2
        return {
            "activation_energy": Ea,
            "pre_exponential_factor": A,
            "r_squared": r2,
        } 