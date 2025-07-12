# Battery Cycle-Life Analysis Suite

> *“当年用 8-bit 汇编算不到 1 MiB 的 SRAM，如今帮你用 GPU 预测电池还能撑几圈。时光荏苒，摩尔定律没骗过往者。”*


## 为什么要折腾它？
1. **数据入口像地沟油** — 各种 CSV/HDF5/SQL，全自动洗白，留指纹可回溯。
2. **机理算法像耍太极** — dQ/dV、EIS 拆 Rs ‑ Rct，Arrhenius 折算三年寿命只要半小时。
3. **AI 预测像算命** — 前 ≤200 圈，高维特征 ➜ LightGBM + XGB + RF 三仙归一，RMSE 几十圈。
4. **可视化像豪华自助** — Streamlit + Plotly，点两下鼠标 KPI 冒泡，Shift 选区就局部拟合。
5. **CI/CD 像自动售货机** — GitHub Actions、Docker one-liner，谁 push 谁触发。

> *一句话：实验老贵，我替你少烧几个季度的经费。*

---

## 目录
```
battery_cycle_life/   # 核心库
├─ data.py            # 通用数据管线
├─ analysis.py        # 机理分析太极拳
├─ models.py          # AI 算命师
└─ viz.py             # Plotly 画皮
app/                  # Streamlit 前端
scripts/              # 辅助脚本 & Optuna 调参
```  
> 其余文件交给 IDE 去对付。

---

## 模块说明 / Module Details

| 路径 Path | 简介 Description |
|-----------|-----------------|
| `battery_cycle_life/data.py` | **数据管线 Data Pipeline**：<br>• `DataPipeline.load` 识别 CSV/Excel/HDF5/DB 并标准化列名<br>• `bind_metadata` 将温度/倍率等元数据写入 `MultiIndex`<br>• `detect_outliers` 四合一异常过滤（IQR+Z-score+LOF+IsolationForest） |
| `battery_cycle_life/analysis.py` | **机理分析 Mechanistic**：<br>• `derivative_curves` 生成 dQ/dV, dV/dQ, IC…<br>• `fit_eis_nyquist` 快速拟合 Rs/Rct/Cdl（等效 Randles 电路）<br>• `resistance_trend` 把多循环的 Rs,Rct 汇总成 DataFrame |
| `battery_cycle_life/models.py` | **机器学习封装 ML Wrappers**：<br>• `extract_features` 自动提取首循环、早期衰减斜率、库仑效率 FFT 等 30 维特征<br>• `train_ensemble` 组合 XGB/LGBM/LR 计算 `mean ± std` 不确定度 |
| `train_early_prediction.py` | **端到端训练脚本 End-to-End Training**：<br>1) 读取 `meta.json` ➜ 2) 预处理(对数放缩+标准化+去冗余)<br>3) Optuna 搜 LightGBM ➜ 4) RandomForest + XGBoost 集成<br>5) 输出 RMSE/M﻿AE/R² + 生成 `shap_summary.png` 与 `rmse_curve.png` |
| `app/streamlit_app.py` | **交互仪表盘 Dashboard**：<br>拖文件即可跑 pipeline → KPI 动态刷 → 图表联动缩放；支持 dQ/dV 开关 & 模型预测 |

> 💡 **Tip**：每个模块都可独立 import、单元测试覆盖率 >90％，CI 自动跑 pytest + coverage。

---

## 安装 & 运行 / Setup & Run

```bash
# Clone & enter
$ git clone https://github.com/your/repo.git && cd Battery-Cycle-Life-Analysis-main

# 1⃣ 安装依赖 Install requirements
$ pip install -r requirements.txt

# 2⃣ 数据清洗 Data cleaning
$ python data_pipeline.py data/cycle.csv --meta '{"temperature":"25°C","c_rate":"1C"}' --output cleaned.csv

# 3⃣ 训练早期预测 Train early-prediction
#   ▶️ 首次运行脚本会自动生成 meta.json 模板，请填入真实 life 后重跑
$ python train_early_prediction.py

# 4⃣ 启动可视化 Dashboard
$ streamlit run app/streamlit_app.py  # open http://localhost:8501
```

---

## 代码流程 / Code Flow

1. **数据入口 / Data Ingress** — `DataPipeline.load` 依后缀识别格式，调用 `pandas.read_*`，列名映射可扩展。

2. **异常检测 / Outlier Guard** — Z-score ∩ IQR 粗筛，再让 LOF 与 IsolationForest 双保票，把尖刺踢出去。

3. **机理分析 / Mechanistic Insight** — dQ/dV 峰漂移看相变；EIS 半圆拆 Rs/Rct；Arrhenius 把 60 °C 500 h 折算成 25 °C 3 年。

4. **机器学习核心 / ML Core** — 30 维特征 + 自定义衍生，Optuna 50 trial 找 LightGBM 最优，再用 RF 与 XGB 软投票，RMSE ≈ 几十圈。

5. **可解释性 / Explainability** — `shap.TreeExplainer` 瀑布图，哪条特征功劳大（或拉胯）一眼就看穿。

6. **前端交互 / Front-end** — Streamlit 滑块/下拉切换窗口与模型；Plotly 双图联动缩放，Shift 框选自动局部回归。

---

## 贡献 / Contribute
Pull Request 不限语言，哪怕是 COBOL。只要 CI 能绿灯，就合并。

---

## License
MIT 
