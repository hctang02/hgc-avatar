
import os
from tqdm import tqdm

# 操作目录
base_dir = "/mnt/ssd2tC/hctang/GPS-Gaussian/new_thuman_sub01"

# 相机编号映射（cam00 到 cam23）
camera_ids = [
    "cam00", "cam01", "cam02", "cam03",
    "cam04", "cam05", "cam06", "cam07",
    "cam08", "cam09", "cam10", "cam11",
    "cam12", "cam13", "cam14", "cam15",
    "cam16", "cam17", "cam18", "cam19",
    "cam20", "cam21", "cam22", "cam23",
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
                camera_id = filename.split(".")[0]  # 提取相机编号（cam00, cam01, ...）
                if camera_id in camera_id_to_index:
                    new_name = f"{camera_id_to_index[camera_id]}.jpg"  # 0.jpg, 1.jpg, ...
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
                parts = filename.split("_")  # 分割文件名（cam00_extrinsic.npy）
                camera_id = parts[0]  # 提取相机编号（cam00）
                if camera_id in camera_id_to_index:
                    new_name = f"{camera_id_to_index[camera_id]}_{parts[1]}"  # 0_extrinsic.npy
                    os.rename(
                        os.path.join(parm_dir, filename),
                        os.path.join(parm_dir, new_name)
                    )

print("重命名完成！")