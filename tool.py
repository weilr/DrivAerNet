"""
    csv_prefix_remove
"""
import os.path

import pandas as pd
from fontTools.subset import subset

# # 读取 CSV
# df = pd.read_csv('AeroCoefficients_DrivAerNet_FilteredCorrected.csv')
#
# # 去掉 "DrivAer_" 前缀
# df['Design'] = df['Design'].str.replace(r'^DrivAer_', '', regex=True)
#
# # 保存
# df.to_csv('AeroCoefficients_DrivAerNet_FilteredCorrected_no_prefix.csv', index=False)

"""
    移动文件夹
"""
# import os
# import shutil
#
#
# def move_stl_files_to_base(base_dir, target_dir):
#     cnt = 1
#     for root, dirs, files in os.walk(base_dir, topdown=False):
#         # 移动文件操作
#         for file in files:
#             if file.endswith('.vtk'):
#                 file_path = os.path.join(root, file)
#                 target_path = os.path.join(target_dir, file)
#
#                 # 目标文件存在时跳过
#                 if os.path.exists(target_path):
#                     print(f"{target_path} existed")
#                     continue
#
#                 # 执行文件移动
#                 print(f"{cnt}: Moving {os.path.relpath(file_path, base_dir)}")
#                 cnt += 1
#                 shutil.move(file_path, target_path)
#
#         # 空文件夹检测与删除
#         try:
#             if not os.listdir(root):  # 判断目录是否为空
#                 os.rmdir(root)
#                 print(f"◆ 已删除空文件夹: {os.path.relpath(root, base_dir)}")
#         except Exception as e:
#             print(f"删除失败: {root} - {str(e)}")
#
#
# if __name__ == '__main__':
#     target_dir = 'Z:/DrivAerNet++/CFD/'
#     base_dir = "Z:/DrivAerNet++_zip/CFD/"
#     move_stl_files_to_base(base_dir, target_dir)


"""
    分割自己的数据集
"""

# import pandas as pd
# from sklearn.model_selection import train_test_split
#
# # 读取CSV文件
# outpath = './train_test_splits'
# file_path = './DrivAerNet_v1/AeroCoefficients_DrivAerNet_FilteredCorrected_no_prefix.csv'
# df = pd.read_csv(file_path)
#
# # 为了训练，去掉一部分数据
# df, _ = train_test_split(df, test_size=0.99, random_state=42)
#
# # 划分训练集、测试集、验证集
# train_data, temp_data = train_test_split(df, test_size=0.4, random_state=42)
# val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
#
# if not os.path.exists(outpath):
#     os.makedirs(outpath)
# # 将Design列保存为txt文件
# train_data[['Design']].to_csv(os.path.join(outpath, 'train_design_ids.txt'), header=False, index=False, sep='\t')
# val_data[['Design']].to_csv(os.path.join(outpath, 'val_design_ids.txt'), header=False, index=False, sep='\t')
# test_data[['Design']].to_csv(os.path.join(outpath, 'test_design_ids.txt'), header=False, index=False, sep='\t')
#
# # 输出分割后文件的信息
# print(f"训练集: {train_data.shape[0]}条数据")
# print(f"验证集: {val_data.shape[0]}条数据")
# print(f"测试集: {test_data.shape[0]}条数据")


"""
    绘制train_loss和test_loss曲线
"""
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 读取 .npy 文件
# train_loss = np.load('DrivAerNet_v1/RegDGCNN/models/CdPrediction_DrivAerNet_20250319_000814_100epochs_5000numPoint_0.4dropout_train_losses.npy')
# test_loss = np.load('DrivAerNet_v1/RegDGCNN/models/CdPrediction_DrivAerNet_20250319_000814_100epochs_5000numPoint_0.4dropout_val_losses.npy')
#
# # 绘制损失曲线
# plt.figure(figsize=(10, 5))
# plt.plot(train_loss, label='Train Loss', linestyle='-', marker='o')
# plt.plot(test_loss, label='Test Loss', linestyle='-', marker='s')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Train and Test Loss Curve')
# plt.legend()
# plt.grid(True)
# plt.show()


"""
    移动文件
"""
#
# import os
# import shutil
#
# # 读取txt文件
# txt_file = 'train_test_splits/val_design_ids.txt'  # 替换为你的txt文件路径
# source_folder = 'D:/Environment/PyCharmProject/DrivAerNet/3DMeshesSTL'  # 替换为源文件夹路径
# destination_folder = 'D:/Environment/PyCharmProject/TestData'  # 替换为目标文件夹路径
#
# # 检查目标文件夹是否存在，如果不存在则创建
# if not os.path.exists(destination_folder):
#     os.makedirs(destination_folder)
#
# # 打开txt文件并逐行读取文件名
# with open(txt_file, 'r', encoding='utf-8') as f:
#     for line in f:
#         # 去除行末的换行符
#         file_name = line.strip()+ ".stl"
#         source_file = os.path.join(source_folder, file_name )
#         destination_file = os.path.join(destination_folder, file_name )
#
#         # 检查源文件是否存在
#         if os.path.exists(source_file):
#             # 复制文件
#             # shutil.copy(source_file, destination_file)
#             print(f"文件 {file_name} 复制成功！")
#         else:
#             print(f"文件 {file_name} 不存在！")


"""
    判断数据集中的文件是否存在
"""
# import os
# import pandas as pd
#
# csv_path = './DrivAerNet_v1/AeroCoefficients_DrivAerNet_FilteredCorrected_no_prefix.csv'  # CSV 文件路径
# data_dir = './3DMeshesSTL'  # 数据集所在文件夹
# filename_column = 'Design'  # CSV 中表示文件名的列名
#
# df = pd.read_csv(csv_path)
#
# # 检查每个文件是否存在
# missing_files = []
#
# for file in df[filename_column]:
#     file_path = os.path.join(data_dir, file + ".stl")
#     if not os.path.isfile(file_path):
#         missing_files.append(file)
#
# # 打印结果
# print(f"共 {len(missing_files)} 个文件缺失：")
# for file in missing_files:
#     print(file)
#
# subset_ids = []
# with open("train_test_splits/test_design_ids.txt", 'r') as file:
#     subset_ids = file.read().split()
# with open("train_test_splits/train_design_ids.txt", 'r') as file:
#     subset_ids += file.read().split()
# with open("train_test_splits/val_design_ids.txt", 'r') as file:
#     subset_ids += file.read().split()
#
# missing_files = []
# for file in subset_ids:
#     file_path = os.path.join(data_dir, file + ".stl")
#     if not os.path.isfile(file_path):
#         missing_files.append(file)
#
# print(f"共 {len(missing_files)} 个文件缺失：")
# for file in missing_files:
#     print(file)


"""
    查看参数量
"""

import torch
from torchinfo import summary
from DrivAerNet_v1.RegDGCNN.model import RegDGCNN
from collections import OrderedDict

# 假设你的模型是 ResNet，下面是一个例子，你可以替换成你的模型
# 先加载模型
config = {
    'exp_name': 'CdPrediction_DrivAerNet',
    'cuda': True,
    'seed': 1,
    'num_points': 5000,
    'lr': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'dropout': 0.4,
    'emb_dims': 512,
    'k': 40,
    'num_workers': 64,
    'optimizer': 'adam',
    # 'channels': [6, 64, 128, 256, 512, 1024],
    # 'linear_sizes': [128, 64, 32, 16],sq
    'dataset_path': os.path.join('3DMeshesSTL'),  # Update this with your dataset path
    'aero_coeff': os.path.join('DrivAerNet_v1',
                               'AeroCoefficients_DrivAerNet_FilteredCorrected_no_prefix.csv'),
    'subset_dir': os.path.join('train_test_splits')
}

# 假设你的模型类是 RegDGCNN
model = RegDGCNN(args=config)  # 先创建模型实例

# 加载 state_dict
state_dict = torch.load(
    './models/CdPrediction_DrivAerNet_20250328_142610_100epochs_5000numPoint_0.4dropout_best_model.pth')
model.load_state_dict(state_dict)

# 打印模型的总结
summary(model, input_size=(config['batch_size'], 3, 5000))  # 修改 input_size 为你的模型输入形状
