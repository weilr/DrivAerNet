"""
    csv_prefix_remove
"""
import os.path

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
# file_path = 'DrivAerNetPlusPlus_Cd_8k_Updated.csv'
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


# """
#     移动文件
# """
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
