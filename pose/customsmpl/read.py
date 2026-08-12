
# import numpy as np

# # 读取 .npy 文件
# npy_file = "samples_00_to_02_smpl_params.npy"
# data = np.load(npy_file, allow_pickle=True)  # 允许包含 Python 对象

# print(type(data))  # 打印数据类型
# print(data)        # 打印数据内容（如果数据很大，可 print(data.keys())）






# import numpy as np

# # 读取 .npz 文件
# # data = np.load('samples_00_to_02_smpl_params.npz', allow_pickle=True)
# data = np.load('output_try.pkl', allow_pickle=True)

# # 打印文件中的所有键（数组名称）
# print("文件中的键:", data.files)

# # 访问具体的数据
# for key in data.files:
#     print(f"键: {key}, 数据类型: {data[key].dtype}, 形状: {data[key].shape}")
#     print(data[key])  # 打印具体数据（如果数据较大，可以选择不打印）

# # 关闭文件（可选，但推荐）
# data.close()











import pickle
import joblib
import numpy as np

# 读取 .pkl 文件
# with open('output_try.pkl', 'rb') as f:
data = joblib.load('output_try.pkl')

# 判断数据类型
if isinstance(data, dict):
    print("文件中的键:", data.keys())

    # 遍历字典中的数据
    for key, value in data.items():
        print(f"键: {key}, 数据类型: {type(value)}")
        if isinstance(value, np.ndarray):
            print(f"形状: {value.shape}, 数据类型: {value.dtype}")
        # print(value[:5])  # 如果数据较大，可以选择 print(value[:10]) 只打印部分数据
else:
    print("数据不是字典类型:", type(data))
    print(data)

