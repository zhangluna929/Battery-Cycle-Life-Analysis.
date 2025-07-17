"""
电池循环寿命分析系统
主程序入口
"""

import streamlit as st
from battery_cycle_life.viz import Dashboard

def main():
    """主函数"""
    # 设置页面
    st.set_page_config(
        page_title="电池循环寿命分析系统",
        page_icon="🔋",
        layout="wide"
    )
    
    # 运行仪表盘
    dashboard = Dashboard()
    dashboard.run()

if __name__ == "__main__":
    main() 