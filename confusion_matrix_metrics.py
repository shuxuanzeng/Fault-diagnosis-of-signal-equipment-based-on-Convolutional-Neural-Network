#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: Fault-diagnosis-of-signal-equipment-based-on-Convolutional-Neural-Network
@File: confusion_matrix_metrics.py
@Author: Generated for signal equipment fault diagnosis
@Date: 2024
@Description: 计算混淆矩阵分类指标的Python代码
              处理240x240的混淆矩阵，计算每个类别的分类指标
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional


class ConfusionMatrixMetrics:
    """
    混淆矩阵分类指标计算类
    
    功能:
    1. 读取混淆矩阵数据（240行×240列）
    2. 计算每个类别的TP, FP, FN, TN
    3. 计算每个类别的Precision, Recall, Accuracy
    4. 计算平均指标
    5. 输出结果到表格和CSV文件
    """
    
    def __init__(self, confusion_matrix: np.ndarray, class_names: Optional[List[str]] = None):
        """
        初始化混淆矩阵指标计算器
        
        Args:
            confusion_matrix: 混淆矩阵数组 (n_classes x n_classes)
            class_names: 类别名称列表，如果为None则自动生成3D_SPK_XXXXX格式
        """
        self.confusion_matrix = np.array(confusion_matrix)
        self.n_classes = self.confusion_matrix.shape[0]
        
        # 验证矩阵是方阵
        if self.confusion_matrix.shape[0] != self.confusion_matrix.shape[1]:
            raise ValueError("混淆矩阵必须是方阵")
        
        # 设置类别名称
        if class_names is None:
            self.class_names = [f"3D_SPK_{str(i).zfill(5)}" for i in range(self.n_classes)]
        else:
            if len(class_names) != self.n_classes:
                raise ValueError("类别名称数量必须与混淆矩阵维度一致")
            self.class_names = class_names
            
        # 初始化结果存储
        self.metrics_df = None
        self.average_metrics = None
    
    def calculate_basic_metrics(self) -> Dict[str, np.ndarray]:
        """
        计算每个类别的基本指标: TP, FP, FN, TN
        
        Returns:
            包含TP, FP, FN, TN数组的字典
        """
        n = self.n_classes
        total_samples = np.sum(self.confusion_matrix)
        
        # 初始化指标数组
        tp = np.zeros(n)  # True Positive
        fp = np.zeros(n)  # False Positive  
        fn = np.zeros(n)  # False Negative
        tn = np.zeros(n)  # True Negative
        
        for i in range(n):
            # TP: 对角线上的值（正确分类的样本数）
            tp[i] = self.confusion_matrix[i, i]
            
            # FP: 该列的总和减去TP（被错误分类为该类的样本数）
            fp[i] = np.sum(self.confusion_matrix[:, i]) - tp[i]
            
            # FN: 该行的总和减去TP（该类被错误分类为其他类的样本数）
            fn[i] = np.sum(self.confusion_matrix[i, :]) - tp[i]
            
            # TN: 总样本数减去TP、FP、FN（正确拒绝的样本数）
            tn[i] = total_samples - tp[i] - fp[i] - fn[i]
        
        return {
            'TP': tp,
            'FP': fp, 
            'FN': fn,
            'TN': tn
        }
    
    def calculate_performance_metrics(self, basic_metrics: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        计算每个类别的性能指标: Precision, Recall, Accuracy
        
        Args:
            basic_metrics: 基本指标字典(TP, FP, FN, TN)
            
        Returns:
            包含Precision, Recall, Accuracy数组的字典
        """
        tp = basic_metrics['TP']
        fp = basic_metrics['FP']
        fn = basic_metrics['FN']
        tn = basic_metrics['TN']
        
        n = self.n_classes
        precision = np.zeros(n)
        recall = np.zeros(n)
        accuracy = np.zeros(n)
        
        for i in range(n):
            # 精确率 (Precision) = TP / (TP + FP)
            # 处理除零情况
            if tp[i] + fp[i] == 0:
                precision[i] = 0.0  # 当没有预测为正类时，精确率为0
            else:
                precision[i] = tp[i] / (tp[i] + fp[i])
            
            # 召回率 (Recall) = TP / (TP + FN)  
            # 处理除零情况
            if tp[i] + fn[i] == 0:
                recall[i] = 0.0  # 当没有真实正类时，召回率为0
            else:
                recall[i] = tp[i] / (tp[i] + fn[i])
            
            # 准确率 (Accuracy) = (TP + TN) / (TP + TN + FP + FN)
            total = tp[i] + tn[i] + fp[i] + fn[i]
            if total == 0:
                accuracy[i] = 0.0
            else:
                accuracy[i] = (tp[i] + tn[i]) / total
        
        return {
            'Precision': precision,
            'Recall': recall,
            'Accuracy': accuracy
        }
    
    def calculate_average_metrics(self, performance_metrics: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        计算所有类别的平均指标
        
        Args:
            performance_metrics: 性能指标字典
            
        Returns:
            平均指标字典
        """
        return {
            'Average_Precision': np.mean(performance_metrics['Precision']),
            'Average_Recall': np.mean(performance_metrics['Recall']),
            'Average_Accuracy': np.mean(performance_metrics['Accuracy'])
        }
    
    def calculate_all_metrics(self) -> pd.DataFrame:
        """
        计算所有指标并返回DataFrame
        
        Returns:
            包含所有指标的DataFrame
        """
        # 计算基本指标
        basic_metrics = self.calculate_basic_metrics()
        
        # 计算性能指标
        performance_metrics = self.calculate_performance_metrics(basic_metrics)
        
        # 计算平均指标
        self.average_metrics = self.calculate_average_metrics(performance_metrics)
        
        # 创建DataFrame
        data = {
            'Class_Name': self.class_names,
            'TP': basic_metrics['TP'].astype(int),
            'FP': basic_metrics['FP'].astype(int),
            'FN': basic_metrics['FN'].astype(int),
            'TN': basic_metrics['TN'].astype(int),
            'Precision': performance_metrics['Precision'],
            'Recall': performance_metrics['Recall'],
            'Accuracy': performance_metrics['Accuracy']
        }
        
        self.metrics_df = pd.DataFrame(data)
        return self.metrics_df
    
    def print_results(self):
        """
        打印详细的计算结果
        """
        if self.metrics_df is None:
            self.calculate_all_metrics()
        
        print("=" * 80)
        print("混淆矩阵分类指标计算结果")
        print("=" * 80)
        print(f"混淆矩阵维度: {self.n_classes} x {self.n_classes}")
        print(f"总样本数: {np.sum(self.confusion_matrix)}")
        print()
        
        # 打印每个类别的详细结果
        print("每个类别的详细指标:")
        print("-" * 80)
        
        # 设置pandas显示选项以显示所有列
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        # 格式化数值显示
        with pd.option_context('display.float_format', '{:.4f}'.format):
            print(self.metrics_df.to_string(index=False))
        
        print()
        print("=" * 80)
        print("平均指标:")
        print("=" * 80)
        for metric, value in self.average_metrics.items():
            print(f"{metric}: {value:.4f}")
        print("=" * 80)
    
    def save_to_csv(self, filename: str = "confusion_matrix_metrics.csv"):
        """
        保存结果到CSV文件
        
        Args:
            filename: 输出文件名
        """
        if self.metrics_df is None:
            self.calculate_all_metrics()
        
        # 保存详细指标
        self.metrics_df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        # 保存平均指标到单独的文件
        avg_filename = filename.replace('.csv', '_averages.csv')
        avg_df = pd.DataFrame(list(self.average_metrics.items()), 
                             columns=['Metric', 'Value'])
        avg_df.to_csv(avg_filename, index=False, encoding='utf-8-sig')
        
        print(f"详细指标已保存到: {filename}")
        print(f"平均指标已保存到: {avg_filename}")


def load_confusion_matrix_from_file(filepath: str) -> np.ndarray:
    """
    从文件加载混淆矩阵
    
    Args:
        filepath: 文件路径 (支持 .csv, .txt, .npy 格式)
        
    Returns:
        混淆矩阵数组
    """
    file_ext = os.path.splitext(filepath)[1].lower()
    
    if file_ext == '.csv':
        return pd.read_csv(filepath, header=None).values
    elif file_ext == '.txt':
        return np.loadtxt(filepath)
    elif file_ext == '.npy':
        return np.load(filepath)
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")


def create_sample_confusion_matrix(n_classes: int = 240) -> np.ndarray:
    """
    创建示例混淆矩阵用于测试
    
    Args:
        n_classes: 类别数量
        
    Returns:
        示例混淆矩阵
    """
    # 创建一个对角占优的混淆矩阵
    np.random.seed(42)  # 固定随机种子便于复现
    
    # 基础混淆矩阵，对角线值较大
    matrix = np.random.randint(0, 10, size=(n_classes, n_classes))
    
    # 增强对角线值，模拟良好的分类效果
    for i in range(n_classes):
        matrix[i, i] += np.random.randint(50, 100)
    
    return matrix


def main():
    """
    主函数：演示混淆矩阵指标计算的使用方法
    """
    print("混淆矩阵分类指标计算工具")
    print("=" * 50)
    
    # 示例1: 使用小规模示例矩阵进行演示
    print("示例1: 使用3x3混淆矩阵演示")
    small_matrix = np.array([
        [85, 3, 2],
        [4, 78, 8], 
        [1, 5, 94]
    ])
    
    small_class_names = ["3D_SPK_00001", "3D_SPK_00002", "3D_SPK_00003"]
    
    calculator_small = ConfusionMatrixMetrics(small_matrix, small_class_names)
    calculator_small.calculate_all_metrics()
    calculator_small.print_results()
    calculator_small.save_to_csv("small_matrix_metrics.csv")
    
    print("\n" + "=" * 80 + "\n")
    
    # 示例2: 创建240x240混淆矩阵进行测试
    print("示例2: 使用240x240混淆矩阵测试")
    large_matrix = create_sample_confusion_matrix(240)
    
    calculator_large = ConfusionMatrixMetrics(large_matrix)
    calculator_large.calculate_all_metrics()
    
    # 对于大矩阵，只显示前5行和后5行
    print("240x240混淆矩阵指标计算完成")
    print("显示前5个和后5个类别的结果:")
    print("-" * 80)
    
    # 显示前5个类别
    print("前5个类别:")
    with pd.option_context('display.float_format', '{:.4f}'.format):
        print(calculator_large.metrics_df.head().to_string(index=False))
    
    print("\n后5个类别:")
    with pd.option_context('display.float_format', '{:.4f}'.format):
        print(calculator_large.metrics_df.tail().to_string(index=False))
    
    print("\n平均指标:")
    print("-" * 40)
    for metric, value in calculator_large.average_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 保存大矩阵结果
    calculator_large.save_to_csv("large_matrix_metrics.csv")
    
    print("\n" + "=" * 80)
    print("演示完成！")
    print("文件已保存:")
    print("- small_matrix_metrics.csv (3x3矩阵详细结果)")
    print("- small_matrix_metrics_averages.csv (3x3矩阵平均指标)")
    print("- large_matrix_metrics.csv (240x240矩阵详细结果)")  
    print("- large_matrix_metrics_averages.csv (240x240矩阵平均指标)")


if __name__ == "__main__":
    main()