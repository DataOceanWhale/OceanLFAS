# OceanLFAS
OceanLFAS (LFAS: Liquidity Forecast Analysis System) 是一套针对资金流动性预测分析的解决方案，可以成为相关任务的工具箱。其中包含以下解耦化的步骤，可以选择性地调用：
* 特征工程
* 数据预处理
* 算法模型
* 预测结果的可视化与分析

该项目的特色是运用了贝叶斯神经网络进行处理。

项目依赖的 Python 包：[zhusuan](https://github.com/thu-ml/zhusuan), pandas, scikit-learn, matplotlib, numpy, tqdm。