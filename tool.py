import pandas as pd


"""
    csv_prefix_remove
"""
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
import os
import shutil

def move_stl_files_to_base(base_dir):
    # 遍历 base_dir 中的所有文件夹和文件
    cnt=1
    for root, dirs, files in os.walk(base_dir, topdown=False):  # topdown=False 先遍历子文件夹
        for file in files:
            if file.endswith('.stl'):  # 只处理 .stl 文件
                file_path = os.path.join(root, file)
                # 目标路径为 base_dir 目录
                target_path = os.path.join(base_dir, file)

                # 如果目标路径上已有同名文件，添加后缀避免覆盖
                if os.path.exists(target_path):
                    continue

                print(f"{cnt}:Moving file: {file_path} -> {target_path}")
                cnt=cnt+1
                # 移动文件到目标文件夹
                shutil.move(file_path, target_path)

        # 删除空文件夹
        # for dir in dirs:
        #     dir_path = os.path.join(root, dir)
        #     try:
        #         os.rmdir(dir_path)  # 删除空文件夹
        #         print(f"Removed empty directory: {dir_path}")
        #     except OSError:
        #         print(f"Failed to remove non-empty directory: {dir_path}")

if __name__ == '__main__':
    base_dir = ""  # 请修改为你的目标文件夹路径
    move_stl_files_to_base(base_dir)
