"""
数据面板模块
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import plotly.graph_objects as go
from ..data import DataLoader, Preprocessor
from ..models import Predictor
from ..analysis import ElectrochemicalAnalyzer
from .plots import Plotter

class Dashboard:
    """交互式数据面板"""
    
    def __init__(self):
        """初始化面板"""
        self.data_loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.analyzer = ElectrochemicalAnalyzer()
        self.plotter = Plotter()
        
    def run(self):
        """运行面板"""
        st.title('电池循环寿命分析系统')
        
        # 侧边栏
        self._create_sidebar()
        
        # 主面板
        tab1, tab2, tab3 = st.tabs(['数据分析', '机理分析', '寿命预测'])
        
        with tab1:
            self._data_analysis_tab()
            
        with tab2:
            self._mechanism_analysis_tab()
            
        with tab3:
            self._life_prediction_tab()
            
    def _create_sidebar(self):
        """创建侧边栏"""
        st.sidebar.title('配置')
        
        # 文件上传
        uploaded_file = st.sidebar.file_uploader(
            "上传数据文件",
            type=['csv', 'xlsx']
        )
        
        if uploaded_file is not None:
            try:
                self.data = self.data_loader.load(uploaded_file)
                st.sidebar.success('数据加载成功！')
                
                # 数据预处理选项
                if st.sidebar.checkbox('数据预处理'):
                    self.data = self.preprocessor.fit_transform(self.data)
                    st.sidebar.success('预处理完成！')
                    
            except Exception as e:
                st.sidebar.error(f'数据加载失败：{str(e)}')
                
    def _data_analysis_tab(self):
        """数据分析标签页"""
        if not hasattr(self, 'data'):
            st.info('请先上传数据文件')
            return
            
        st.header('数据概览')
        
        # 数据统计
        col1, col2 = st.columns(2)
        with col1:
            st.write('数据形状：', self.data.shape)
        with col2:
            st.write('数据列：', list(self.data.columns))
            
        # 数据预览
        st.subheader('数据预览')
        st.dataframe(self.data.head())
        
        # 基本统计量
        st.subheader('基本统计量')
        st.dataframe(self.data.describe())
        
        # 相关性分析
        st.subheader('相关性分析')
        corr = self.data.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu'
        ))
        st.plotly_chart(fig)
        
    def _mechanism_analysis_tab(self):
        """机理分析标签页"""
        if not hasattr(self, 'data'):
            st.info('请先上传数据文件')
            return
            
        st.header('电化学分析')
        
        # 分析类型选择
        analysis_type = st.selectbox(
            '选择分析类型',
            ['容量衰减分析', 'dQ/dV分析', '阻抗分析']
        )
        
        if analysis_type == '容量衰减分析':
            self._capacity_fade_analysis()
        elif analysis_type == 'dQ/dV分析':
            self._dqdv_analysis()
        elif analysis_type == '阻抗分析':
            self._impedance_analysis()
            
    def _capacity_fade_analysis(self):
        """容量衰减分析"""
        if 'cycle' not in self.data.columns or 'capacity' not in self.data.columns:
            st.error('数据缺少必要的列（cycle/capacity）')
            return
            
        fig = self.plotter.plot_capacity_fade(
            self.data['cycle'].values,
            self.data['capacity'].values,
            show_fit=st.checkbox('显示拟合曲线', value=True)
        )
        st.pyplot(fig)
        
    def _dqdv_analysis(self):
        """dQ/dV分析"""
        if 'voltage' not in self.data.columns or 'current' not in self.data.columns:
            st.error('数据缺少必要的列（voltage/current）')
            return
            
        # 选择循环
        cycle = st.selectbox('选择循环', self.data['cycle'].unique())
        cycle_data = self.data[self.data['cycle'] == cycle]
        
        # 计算dQ/dV
        result = self.analyzer.analyze_dqdv(
            cycle_data['voltage'].values,
            cycle_data['current'].values,
            cycle_data.index.values
        )
        
        fig = self.plotter.plot_dqdv(
            result['voltage'],
            result['dqdv'],
            result['peaks']
        )
        st.pyplot(fig)
        
    def _impedance_analysis(self):
        """阻抗分析"""
        if 'z_real' not in self.data.columns or 'z_imag' not in self.data.columns:
            st.error('数据缺少必要的列（z_real/z_imag）')
            return
            
        # 拟合EIS
        result = self.analyzer.analyze_eis(
            self.data['z_real'].values,
            self.data['z_imag'].values
        )
        
        fig = self.plotter.plot_eis(
            self.data['z_real'].values,
            self.data['z_imag'].values,
            result
        )
        st.pyplot(fig)
        
    def _life_prediction_tab(self):
        """寿命预测标签页"""
        if not hasattr(self, 'data'):
            st.info('请先上传数据文件')
            return
            
        st.header('寿命预测')
        
        # 模型选择
        model_type = st.selectbox(
            '选择模型类型',
            ['基础模型', '不确定度模型']
        )
        
        # 特征选择
        feature_cols = st.multiselect(
            '选择特征',
            list(self.data.columns),
            default=['capacity', 'voltage', 'current']
        )
        
        if st.button('开始预测'):
            try:
                # 加载预训练模型（示例）
                predictor = Predictor(None)  # 实际应用中需要加载真实模型
                
                # 预测
                features = self.data[feature_cols].values
                if model_type == '基础模型':
                    predictions = predictor.predict(features)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=predictions,
                        mode='lines+markers',
                        name='预测寿命'
                    ))
                    st.plotly_chart(fig)
                    
                else:  # 不确定度模型
                    predictions, uncertainties = predictor.predict(
                        features,
                        return_uncertainty=True
                    )
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=predictions,
                        mode='lines',
                        name='预测值'
                    ))
                    fig.add_trace(go.Scatter(
                        y=predictions + 2*uncertainties,
                        mode='lines',
                        line=dict(dash='dash'),
                        name='95%置信区间'
                    ))
                    fig.add_trace(go.Scatter(
                        y=predictions - 2*uncertainties,
                        mode='lines',
                        line=dict(dash='dash'),
                        showlegend=False
                    ))
                    st.plotly_chart(fig)
                    
            except Exception as e:
                st.error(f'预测失败：{str(e)}') 