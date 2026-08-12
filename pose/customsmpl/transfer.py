import numpy as np
from smplx import SMPL, SMPLX

# 加载 SMPL 模型
smpl_model = SMPL(model_path='basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')

# 加载 SMPL-X 模型
smplx_model = SMPLX(model_path='SMPLX_NEUTRAL.npz')

# 读取 poses_optimized.npz 文件
data = np.load('poses_optimized.npz', allow_pickle=True)

# 获取 body_pose 数据
body_pose_smpl = data['body_pose']  # 形状: [N, 69]

# # SMPL 关节索引到 SMPL-X 关节索引的映射
# smpl_to_smplx_joint_indices = [
#     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
# ]

# # 将 SMPL 的 body_pose 映射到 SMPL-X 的 body_pose
# body_pose_smplx = np.zeros((body_pose_smpl.shape[0], 63), dtype=np.float32)
# for i, joint_idx in enumerate(smpl_to_smplx_joint_indices):
#     body_pose_smplx[:, i*3:(i+1)*3] = body_pose_smpl[:, joint_idx*3:(joint_idx+1)*3]

if body_pose_smpl.shape[1] == 69:
    body_pose_smplx = body_pose_smpl[:, :63]  # 只保留前 63 个参数

# 更新数据字典
new_data = dict(data)  # 复制原始数据
new_data['body_pose'] = body_pose_smplx  # 替换为转换后的 body_pose

# 保存为新的 new_poses_optimized.npz 文件l
np.savez('new_poses_optimized.npz', **new_data)

print("转换完成，文件已保存为 new_poses_optimized.npz")
