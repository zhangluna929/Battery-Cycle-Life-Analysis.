"""
结构分析模块
提供 XRD 峰提取、晶格常数计算及形貌统计分析
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.signal import find_peaks

__all__ = ["StructuralAnalyzer"]

class StructuralAnalyzer:
    """用于结构表征分析的工具类"""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # XRD Analysis
    # ------------------------------------------------------------------
    def analyze_xrd(
        self,
        two_theta: np.ndarray,
        intensity: np.ndarray,
        wavelength: float = 1.5406,  # Cu Kα (Å)
        prominence: float | None = None,
    ) -> Dict:
        """简单的 XRD 峰检索与 d-spacing 计算。

        返回衍射峰信息与估算的晶格常数 (假定立方晶系)。
        """
        if prominence is None:
            prominence = 0.05 * intensity.max()
        peaks, props = find_peaks(intensity, prominence=prominence)
        d_spacing = wavelength / (2 * np.sin(np.radians(two_theta[peaks] / 2)))

        # 猜测前三个峰对应 (111)(200)(220)
        hkl = [(1,1,1), (2,0,0), (2,2,0)]
        a_values: List[float] = []
        for idx, (h,k,l) in enumerate(hkl):
            if idx >= len(d_spacing):
                break
            a = d_spacing[idx] * np.sqrt(h**2 + k**2 + l**2)
            a_values.append(a)
        lattice_const = float(np.mean(a_values)) if a_values else None
        return {
            "peaks_index": peaks,
            "two_theta_peaks": two_theta[peaks],
            "peak_intensity": intensity[peaks],
            "d_spacing": d_spacing,
            "estimated_a": lattice_const,
        }

    # ------------------------------------------------------------------
    # Morphology Analysis
    # ------------------------------------------------------------------
    def analyze_morphology(
        self,
        particle_sizes: np.ndarray,
        aspect_ratios: Optional[np.ndarray] = None,
        bins: int | str = "auto",
    ) -> Dict:
        """颗粒尺寸 / 形貌统计。"""
        size_hist = np.histogram(particle_sizes, bins=bins)
        result: Dict = {
            "mean_size": float(np.mean(particle_sizes)),
            "std_size": float(np.std(particle_sizes)),
            "hist_size": size_hist,
        }
        if aspect_ratios is not None:
            ar_hist = np.histogram(aspect_ratios, bins=bins)
            result.update(
                {
                    "mean_aspect": float(np.mean(aspect_ratios)),
                    "std_aspect": float(np.std(aspect_ratios)),
                    "hist_aspect": ar_hist,
                }
            )
        return result 