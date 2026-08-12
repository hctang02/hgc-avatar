import os
from tqdm import tqdm

# 操作目录
base_dir = "/mnt/ssd2tC/hctang/GPS-Gaussian/new_avatarrex_lbn1"

# 相机编号映射
camera_ids = [
    "22010708", "22010710", "22010714", "22010716",
    "22053903", "22053907", "22053908", "22053912",
    "22053917", "22053923", "22053925", "22053926",
    "22070928", "22070932", "22070935", "22139907"
]
camera_id_to_index = {camera_id: str(i) for i, camera_id in enumerate(camera_ids)}

# 遍历所有帧（0000到0500）
for frame in tqdm(range(501), desc="Renaming files"):
    frame_str = f"{frame:04d}"  # 格式化为 0000, 0001, ..., 0500

    # 1. 重命名 img 文件夹中的文件
    img_dir = os.path.join(base_dir, "img", frame_str)
    if os.path.exists(img_dir):
        for filename in os.listdir(img_dir):
            if filename.endswith(".jpg"):
                camera_id = filename.split(".")[0]  # 提取相机编号
                if camera_id in camera_id_to_index:
                    new_name = f"{camera_id_to_index[camera_id]}.jpg"
                    os.rename(
                        os.path.join(img_dir, filename),
                        os.path.join(img_dir, new_name)
                    )

    # 2. 重命名 mask 文件夹中的文件
    mask_dir = os.path.join(base_dir, "mask", frame_str)
    if os.path.exists(mask_dir):
        for filename in os.listdir(mask_dir):
            if filename.endswith(".jpg"):
                camera_id = filename.split(".")[0]  # 提取相机编号
                if camera_id in camera_id_to_index:
                    new_name = f"{camera_id_to_index[camera_id]}.jpg"
                    os.rename(
                        os.path.join(mask_dir, filename),
                        os.path.join(mask_dir, new_name)
                    )

    # 3. 重命名 parm 文件夹中的文件
    parm_dir = os.path.join(base_dir, "parm", frame_str)
    if os.path.exists(parm_dir):
        for filename in os.listdir(parm_dir):
            if filename.endswith(".npy"):
                parts = filename.split("_")  # 分割文件名
                camera_id = parts[0]  # 提取相机编号
                if camera_id in camera_id_to_index:
                    new_name = f"{camera_id_to_index[camera_id]}_{parts[1]}"
                    os.rename(
                        os.path.join(parm_dir, filename),
                        os.path.join(parm_dir, new_name)
                    )

print("重命名完成！")