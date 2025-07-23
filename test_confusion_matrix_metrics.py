#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: Fault-diagnosis-of-signal-equipment-based-on-Convolutional-Neural-Network
@File: test_confusion_matrix_metrics.py
@Author: Generated for signal equipment fault diagnosis
@Date: 2024
@Description: 测试混淆矩阵分类指标计算的示例代码
"""

import numpy as np
import pandas as pd
from confusion_matrix_metrics import ConfusionMatrixMetrics, load_confusion_matrix_from_file, create_sample_confusion_matrix


def test_custom_confusion_matrix():
    """
    测试自定义混淆矩阵的指标计算
    """
    print("测试自定义混淆矩阵指标计算")
    print("=" * 50)
    
    # 创建一个5x5的示例混淆矩阵
    custom_matrix = np.array([
        [45, 2, 1, 1, 1],
        [3, 42, 3, 1, 1], 
        [1, 2, 44, 2, 1],
        [2, 1, 1, 43, 3],
        [1, 1, 2, 2, 44]
    ])
    
    # 自定义类别名称
    custom_class_names = [
        "3D_SPK_Motor1",
        "3D_SPK_Motor2", 
        "3D_SPK_Sensor1",
        "3D_SPK_Sensor2",
        "3D_SPK_Normal"
    ]
    
    # 创建计算器并计算指标
    calculator = ConfusionMatrixMetrics(custom_matrix, custom_class_names)
    calculator.calculate_all_metrics()
    calculator.print_results()
    calculator.save_to_csv("custom_matrix_metrics.csv")
    
    return calculator


def test_file_loading():
    """
    测试从文件加载混淆矩阵
    """
    print("测试从文件加载混淆矩阵")
    print("=" * 50)
    
    # 创建示例矩阵并保存为不同格式
    sample_matrix = create_sample_confusion_matrix(5)
    
    # 保存为CSV格式
    np.savetxt("sample_matrix.csv", sample_matrix, delimiter=",", fmt="%d")
    
    # 保存为TXT格式  
    np.savetxt("sample_matrix.txt", sample_matrix, fmt="%d")
    
    # 保存为NPY格式
    np.save("sample_matrix.npy", sample_matrix)
    
    print("已创建示例文件：")
    print("- sample_matrix.csv")
    print("- sample_matrix.txt") 
    print("- sample_matrix.npy")
    print()
    
    # 测试从CSV加载
    print("从CSV文件加载：")
    matrix_from_csv = load_confusion_matrix_from_file("sample_matrix.csv")
    calculator_csv = ConfusionMatrixMetrics(matrix_from_csv)
    metrics_csv = calculator_csv.calculate_all_metrics()
    print(f"矩阵维度: {matrix_from_csv.shape}")
    print(f"前3个类别的精确率: {metrics_csv['Precision'].head(3).tolist()}")
    print()
    
    # 测试从TXT加载
    print("从TXT文件加载：")
    matrix_from_txt = load_confusion_matrix_from_file("sample_matrix.txt")
    calculator_txt = ConfusionMatrixMetrics(matrix_from_txt)
    metrics_txt = calculator_txt.calculate_all_metrics()
    print(f"矩阵维度: {matrix_from_txt.shape}")
    print(f"前3个类别的召回率: {metrics_txt['Recall'].head(3).tolist()}")
    print()
    
    # 测试从NPY加载
    print("从NPY文件加载：")
    matrix_from_npy = load_confusion_matrix_from_file("sample_matrix.npy")
    calculator_npy = ConfusionMatrixMetrics(matrix_from_npy)
    metrics_npy = calculator_npy.calculate_all_metrics()
    print(f"矩阵维度: {matrix_from_npy.shape}")
    print(f"前3个类别的准确率: {metrics_npy['Accuracy'].head(3).tolist()}")


def test_edge_cases():
    """
    测试边界情况和异常处理
    """
    print("测试边界情况")
    print("=" * 50)
    
    # 测试包含零的混淆矩阵（处理除零情况）
    zero_matrix = np.array([
        [10, 0, 0],
        [0, 0, 5],  # 第二类没有TP，会导致除零
        [0, 3, 15]
    ])
    
    print("测试包含零值的混淆矩阵：")
    calculator_zero = ConfusionMatrixMetrics(zero_matrix)
    calculator_zero.calculate_all_metrics()
    calculator_zero.print_results()
    
    print()
    print("注意：第二个类别的TP=0，Precision和Recall被正确处理为0.0")


def main():
    """
    运行所有测试
    """
    print("混淆矩阵指标计算 - 综合测试")
    print("=" * 60)
    print()
    
    # 测试1：自定义混淆矩阵
    test_custom_confusion_matrix()
    print("\n" + "=" * 60 + "\n")
    
    # 测试2：文件加载
    test_file_loading()
    print("\n" + "=" * 60 + "\n")
    
    # 测试3：边界情况
    test_edge_cases()
    print("\n" + "=" * 60 + "\n")
    
    print("所有测试完成！")
    print("\n生成的文件：")
    import os
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    other_files = [f for f in os.listdir('.') if f.endswith(('.txt', '.npy'))]
    
    for f in csv_files:
        print(f"- {f}")
    for f in other_files:
        print(f"- {f}")


if __name__ == "__main__":
    main()