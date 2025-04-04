"""
    csv_prefix_remove
"""
import os.path

import pandas as pd
from fontTools.subset import subset

from DeepSurrogates.DrivAerNetDataset import DrivAerNetDataset

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
    按照比例分割数据集
"""

# import pandas as pd
# from sklearn.model_selection import train_test_split
#
# # 读取CSV文件
# outpath = './train_splits'
# file_path = './DrivAerNetPlusPlus_Cd_8k_Updated.csv'
# df = pd.read_csv(file_path)
#
# # 为了训练，去掉一部分数据
# # df, _ = train_test_split(df, test_size=0, random_state=42)
#
# # 划分训练集、测试集、验证集
# train_data, temp_data = train_test_split(df, test_size=0.7, random_state=42)
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
    按照数量分割数据集
"""

# import pandas as pd
# import os
# import random
#
# # ======== 配置项 ========
# csv_path = 'DrivAerNetPlusPlus_Cd_8k_Frontal_Area.csv'  # CSV路径，含有Design列
# stl_dir = './3DMeshesSTL'      # 存放STL文件的文件夹
# output_dir = './splits/Frontal_Area_splits2800_600_600'        # 输出划分结果目录
#
# train_size, val_size, test_size = 2800, 600, 600
# stl_suffix = '.stl'            # 文件后缀，可改为 .obj 等
#
# # ======== 创建输出文件夹 ========
# os.makedirs(output_dir, exist_ok=True)
#
# # ======== 读取数据并过滤存在的文件 ========
# df = pd.read_csv(csv_path)
# all_designs = df['Design'].dropna().unique().tolist()
#
# # 过滤存在的STL文件
# valid_designs = [d for d in all_designs if os.path.isfile(os.path.join(stl_dir, f'{d}{stl_suffix}'))]
# print(f"总共 {len(all_designs)} 个 Design，其中 {len(valid_designs)} 个有对应的 STL 文件")
#
# # ======== 随机划分数据 ========
# random.shuffle(valid_designs)
#
# train_ids = valid_designs[:train_size]
# val_ids = valid_designs[train_size:train_size + val_size]
# test_ids = valid_designs[train_size + val_size:train_size + val_size + test_size]
#
# # ======== 保存结果 ========
# def save_ids(ids, name):
#     with open(os.path.join(output_dir, f'{name}_design_ids.txt'), 'w') as f:
#         f.write('\n'.join(ids))
#
# save_ids(train_ids, 'train')
# save_ids(val_ids, 'val')
# save_ids(test_ids, 'test')
#
# print(f"[Done] 训练: {len(train_ids)}，验证: {len(val_ids)}，测试: {len(test_ids)}")


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

# import torch
# from torchinfo import summary
# from DrivAerNet_v1.RegDGCNN.model import RegDGCNN
# from collections import OrderedDict
#
# # 假设你的模型是 ResNet，下面是一个例子，你可以替换成你的模型
# # 先加载模型
# config = {
#     'exp_name': 'CdPrediction_DrivAerNet',
#     'cuda': True,
#     'seed': 1,
#     'num_points': 5000,
#     'lr': 0.001,
#     'batch_size': 32,
#     'epochs': 100,
#     'dropout': 0.4,
#     'emb_dims': 512,
#     'k': 40,
#     'num_workers': 64,
#     'optimizer': 'adam',
#     # 'channels': [6, 64, 128, 256, 512, 1024],
#     # 'linear_sizes': [128, 64, 32, 16],sq
#     'dataset_path': os.path.join('3DMeshesSTL'),  # Update this with your dataset path
#     'aero_coeff': os.path.join('DrivAerNetPlusPlus_Cd_8k_Updated.csv'),
#     'subset_dir': os.path.join('splits')
# }
# device = torch.device("cuda" if torch.cuda.is_available() and config['cuda'] else "cpu")
#
# # 加载训练好的模型权重
# model_path = './models/CdPrediction_DrivAerNet_20250330_194316_100epochs_5000numPoint_0.4dropout_best_model.pth'
#
# model = RegDGCNN(args=config).to(device)  # Initialize a new model instance
# if config['cuda'] and torch.cuda.device_count() > 1:
#     device_cnt = torch.cuda.device_count()
#     model = torch.nn.DataParallel(model, device_ids=list(range(device_cnt)))
# model.load_state_dict(torch.load(model_path))
#
# # 打印模型的总结
# summary(model, input_size=(config['batch_size'], 3, config['num_points']))


"""
    合并两个 CSV 文件，并只保留两个文件中共有的项。
"""

# import pandas as pd
# import os
#
#
# def merge_and_check_files(file_path1, file_path2, folder_path, output_file):
#     """
#     合并两个 CSV 文件，并只保留 'Design' 列在两个文件中都存在，
#     且在指定文件夹中也存在对应文件的项。
#
#     Args:
#         file_path1 (str): 第一个 CSV 文件的路径。
#         file_path2 (str): 第二个 CSV 文件的路径。
#         folder_path (str): 包含与 'Design' 列值对应的文件的文件夹路径。
#         output_file (str): 合并后输出的 CSV 文件路径。
#     """
#     try:
#         # 读取两个 CSV 文件到 pandas DataFrame
#         df1 = pd.read_csv(file_path1)
#         df2 = pd.read_csv(file_path2)
#
#         # 使用 'Design' 列作为键进行内连接 (inner join)
#         merged_df = pd.merge(df1, df2, on='Design', how='inner').copy()
#         merged_df.loc[:, 'Cd'] = merged_df['Frontal Area (m²)'] * merged_df['Average Cd']
#
#         # 获取文件夹中所有文件名（不包含扩展名）
#         existing_files = set()
#         for filename in os.listdir(folder_path):
#             # 移除文件扩展名，假设你的文件都有扩展名，并且你想匹配文件名（不含扩展名）
#             name, ext = os.path.splitext(filename)
#             existing_files.add(name)
#
#         # 过滤 merged_df，只保留 'Design' 列值在 existing_files 集合中的行
#         filtered_df = merged_df[merged_df['Design'].isin(existing_files)]
#         # 将过滤后的 DataFrame 保存到新的 CSV 文件
#         filtered_df.to_csv(output_file, index=False, encoding='utf-8-sig')
#
#         print(f"成功合并并检查文件，结果已保存到: {output_file}")
#
#     except FileNotFoundError:
#         print("错误: 找不到指定的文件路径。")
#     except NotADirectoryError:
#         print("错误: 指定的文件夹路径无效。")
#     except Exception as e:
#         print(f"发生错误: {e}")
#
#
# # 替换为你的实际文件路径、文件夹路径和输出文件路径
# file1_path = 'DrivAerNetPlusPlus_CarDesign_Areas.csv'  # 请替换为你的第一个文件路径
# file2_path = 'DrivAerNetPlusPlus_Cd_8k_Updated.csv'  # 请替换为你的第二个文件路径
# folder_path = './3DMeshesSTL'  # 请替换为包含对应文件的文件夹路径
# output_file_path = 'DrivAerNetPlusPlus_Cd_8k_Frontal_Area.csv'  # 请替换为你想要保存的输出文件路径
# # 调用函数执行合并和文件检查操作
# merge_and_check_files(file1_path, file2_path, folder_path, output_file_path)


"""
    缓存数据
"""
# import platform
#
# if platform.system() == "Windows":
#     proj_path = os.getcwd()
# else:
#     proj_path = os.getcwd()
# os.chdir(os.getcwd())
#
# config = {
#     'exp_name': 'CdPrediction_DrivAerNet_Unnormalization',
#     'train_target': 'Cd',
#     'cuda': True,
#     'seed': 1,
#     'num_points': 10000,
#     'lr': 0.001,
#     'batch_size': 32,
#     'epochs': 100,
#     'dropout': 0.4,
#     'emb_dims': 512,
#     'k': 40,
#     'num_workers': 64,
#     'optimizer': 'adam',
#     # 'channels': [6, 64, 128, 256, 512, 1024],
#     # 'linear_sizes': [128, 64, 32, 16],
#     'dataset_path': os.path.join(proj_path, '3DMeshesSTL'),  # Update this with your dataset path
#     'aero_coeff': os.path.join(proj_path, 'DrivAerNetPlusPlus_Cd_8k_Frontal_Area.csv'),
#     'subset_dir': os.path.join(proj_path, 'splits', 'Frontal_Area_splits5600_1200_1200')
# }
#
# dataset = DrivAerNetDataset(root_dir=config['dataset_path'], csv_file=config['aero_coeff'],
#                             num_points=config['num_points'], target=config['train_target'])
#
#
# dataset.generate_cache()