import os
from PIL import Image
from tqdm import tqdm

# 操作目录
mask_dir = "/mnt/ssd2tC/hctang/GPS-Gaussian/new_thuman_sub01/mask"

# 遍历所有帧（0000到0500）
for frame in tqdm(range(501), desc="Converting images"):
    frame_str = f"{frame:04d}"  # 格式化为 0000, 0001, ..., 0500

    # 当前帧的文件夹路径
    frame_folder = os.path.join(mask_dir, frame_str)
    if not os.path.exists(frame_folder):
        continue  # 如果文件夹不存在，跳过

    # 遍历文件夹中的所有 .jpg 文件
    for filename in os.listdir(frame_folder):
        if filename.endswith(".jpg"):
            # 原始文件路径
            jpg_path = os.path.join(frame_folder, filename)
            # 新文件路径（将 .jpg 替换为 .png）
            png_path = os.path.join(frame_folder, filename.replace(".jpg", ".png"))

            # 打开图片并保存为 .png 格式
            with Image.open(jpg_path) as img:
                img.save(png_path, "PNG")

            # 删除原始的 .jpg 文件
            os.remove(jpg_path)

print("转换完成！")