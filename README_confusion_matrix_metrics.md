# 混淆矩阵分类指标计算工具

## 概述

本工具用于计算混淆矩阵的详细分类指标，支持处理240x240或任意大小的混淆矩阵，计算每个类别的精确率、召回率、准确率等指标。

## 功能特性

### 🎯 核心功能
- **混淆矩阵处理**: 支持读取任意大小的混淆矩阵（特别优化了240x240）
- **基本指标计算**: 计算每个类别的TP、FP、FN、TN
- **性能指标计算**: 计算每个类别的精确率(Precision)、召回率(Recall)、准确率(Accuracy)
- **平均指标**: 计算所有类别的平均精确率、召回率、准确率
- **异常处理**: 自动处理除零情况，确保计算稳定性
- **多格式支持**: 支持CSV、TXT、NPY格式的矩阵文件读取
- **结果导出**: 支持将结果保存为CSV文件

### 📊 计算公式

#### 基本指标
- **TP (True Positive)**: 对角线上的值（正确分类的样本数）
- **FP (False Positive)**: 该列的总和减去TP（被错误分类为该类的样本数）
- **FN (False Negative)**: 该行的总和减去TP（该类被错误分类为其他类的样本数）
- **TN (True Negative)**: 总样本数减去TP、FP、FN（正确拒绝的样本数）

#### 性能指标
- **精确率 (Precision)** = TP / (TP + FP)
- **召回率 (Recall)** = TP / (TP + FN)
- **准确率 (Accuracy)** = (TP + TN) / (TP + TN + FP + FN)

## 文件说明

### 核心文件
- `confusion_matrix_metrics.py` - 主要的混淆矩阵指标计算模块
- `test_confusion_matrix_metrics.py` - 测试和示例代码

### 生成文件
运行后会生成以下文件：
- `*_metrics.csv` - 详细的每个类别指标
- `*_metrics_averages.csv` - 平均指标汇总

## 使用方法

### 1. 基本使用（Python代码中）

```python
from confusion_matrix_metrics import ConfusionMatrixMetrics
import numpy as np

# 创建或加载混淆矩阵
confusion_matrix = np.array([
    [85, 3, 2],
    [4, 78, 8], 
    [1, 5, 94]
])

# 可选：自定义类别名称
class_names = ["3D_SPK_00001", "3D_SPK_00002", "3D_SPK_00003"]

# 创建计算器
calculator = ConfusionMatrixMetrics(confusion_matrix, class_names)

# 计算所有指标
metrics_df = calculator.calculate_all_metrics()

# 显示结果
calculator.print_results()

# 保存结果
calculator.save_to_csv("my_metrics.csv")
```

### 2. 从文件加载混淆矩阵

```python
from confusion_matrix_metrics import load_confusion_matrix_from_file, ConfusionMatrixMetrics

# 从文件加载（支持.csv, .txt, .npy格式）
matrix = load_confusion_matrix_from_file("my_confusion_matrix.csv")

# 创建计算器并计算
calculator = ConfusionMatrixMetrics(matrix)
calculator.calculate_all_metrics()
calculator.print_results()
calculator.save_to_csv("results.csv")
```

### 3. 直接运行脚本

```bash
# 运行主脚本，查看示例
python confusion_matrix_metrics.py

# 运行测试脚本，查看更多示例
python test_confusion_matrix_metrics.py
```

## 输出示例

### 控制台输出
```
================================================================================
混淆矩阵分类指标计算结果
================================================================================
混淆矩阵维度: 3 x 3
总样本数: 280

每个类别的详细指标:
--------------------------------------------------------------------------------
  Class_Name  TP  FP  FN  TN  Precision  Recall  Accuracy
3D_SPK_00001  85   5   5 185     0.9444  0.9444    0.9643
3D_SPK_00002  78   8  12 182     0.9070  0.8667    0.9286
3D_SPK_00003  94  10   6 170     0.9038  0.9400    0.9429

================================================================================
平均指标:
================================================================================
Average_Precision: 0.9184
Average_Recall: 0.9170
Average_Accuracy: 0.9452
================================================================================
```

### CSV文件输出
详细指标文件 (`metrics.csv`):
```csv
Class_Name,TP,FP,FN,TN,Precision,Recall,Accuracy
3D_SPK_00001,85,5,5,185,0.9444,0.9444,0.9643
3D_SPK_00002,78,8,12,182,0.9070,0.8667,0.9286
3D_SPK_00003,94,10,6,170,0.9038,0.9400,0.9429
```

平均指标文件 (`metrics_averages.csv`):
```csv
Metric,Value
Average_Precision,0.9184
Average_Recall,0.9170
Average_Accuracy,0.9452
```

## 依赖要求

```bash
pip install pandas numpy
```

## 类别名称格式

- 默认格式：`3D_SPK_XXXXX`（XXXXX为5位数字编号）
- 支持自定义类别名称
- 对于240类别，自动生成从`3D_SPK_00000`到`3D_SPK_00239`

## 特殊情况处理

### 除零处理
当遇到除零情况时，工具会自动处理：
- 如果 TP + FP = 0：精确率设为 0.0
- 如果 TP + FN = 0：召回率设为 0.0
- 如果总样本为 0：准确率设为 0.0

### 大矩阵处理
对于240x240等大型混淆矩阵：
- 优化内存使用
- 支持部分结果显示（前N个和后N个类别）
- 完整结果保存到CSV文件

## 示例场景

### 信号设备故障诊断
本工具特别适用于信号设备故障诊断场景，可以：
1. 评估CNN模型在多类别故障分类上的性能
2. 识别容易混淆的故障类型
3. 为模型优化提供详细的性能分析

### 使用建议
1. **模型评估**: 用于评估训练好的CNN模型在测试集上的表现
2. **错误分析**: 通过混淆矩阵识别模型的薄弱环节
3. **类别平衡**: 检查各类别的precision和recall是否平衡
4. **阈值调优**: 基于精确率-召回率权衡选择最优决策阈值

## 技术说明

- **编程语言**: Python 3.6+
- **核心库**: pandas, numpy
- **编码**: UTF-8 (支持中文)
- **输出格式**: 控制台表格 + CSV文件
- **性能**: 优化的大矩阵处理算法

## 贡献与反馈

如有问题或建议，欢迎提出Issue或Pull Request。