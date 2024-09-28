import os
import json

# 设置目录路径
data_dir = 'GaitData/train'  # 你的数据目录

# 获取目录中的所有文件夹名，并按数字顺序排序
all_folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])

# 生成所需的格式，将所有文件夹划分到训练集
data_split = {
    "TRAIN_SET": all_folders,
    "TEST_SET": []  # 空的测试集
}

# 将结果保存为 JSON 文件
with open('CASIA-B.json', 'w') as json_file:
    json.dump(data_split, json_file, indent=4)

print("所有文件夹已划分到训练集，并保存为 CASIA-B.json")