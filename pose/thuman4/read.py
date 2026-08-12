import numpy as np

# 读取 npz 文件
data = np.load('pose_00.npz')

# 打印所有的 keys
print("Keys:", data.files)

# 遍历 keys，打印对应 value 的格式
for key in data.files:
    value = data[key]
    print(f"Key: {key}, Shape: {value.shape}, Dtype: {value.dtype}")
