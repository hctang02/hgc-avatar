#用于构建相机内外参数的npy文件
import json
import numpy as np
import os

# 定义输出文件夹
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)  # 创建输出文件夹（如果不存在）

# 读取 calibration_full.json 文件
with open("calibration_full.json", "r") as f:
    calibration_data = json.load(f)

# 遍历每个相机的参数
for camera_id, params in calibration_data.items():
    # 提取内参矩阵 K
    K = np.array(params["K"]).reshape(3, 3)
    
    # 提取旋转矩阵 R 和平移向量 T
    R = np.array(params["R"]).reshape(3, 3)
    T = np.array(params["T"]).reshape(3, 1)
    
    # 构建外参矩阵 [R | T]
    extrinsic = np.hstack((R, T))
    
    # 保存内参和外参为 .npy 文件
    intrinsic_path = os.path.join(output_dir, f"{camera_id}_intrinsic.npy")
    extrinsic_path = os.path.join(output_dir, f"{camera_id}_extrinsic.npy")
    
    np.save(intrinsic_path, K)
    np.save(extrinsic_path, extrinsic)

    print(f"已保存 {intrinsic_path} 和 {extrinsic_path}")
