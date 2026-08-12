# #用于把所有的avatarrex的数据处理成GPS-Gaussian的数据格式
# import os
# import shutil
# from tqdm import tqdm

# # 输入路径
# input_root = "/mnt/ssd2tB/haocheng/AnimatableGaussians/data/thuman4.0/subject01"

# # 输出路径
# output_root = "/mnt/ssd2tC/hctang/GPS-Gaussian/new_thuman_sub01"
# img_output_dir = os.path.join(output_root, "img")
# mask_output_dir = os.path.join(output_root, "mask")

# # 创建输出文件夹
# os.makedirs(img_output_dir, exist_ok=True)
# os.makedirs(mask_output_dir, exist_ok=True)

# # 显式列出所有相机视角文件夹
# camera_ids = [
#     "cam00", "cam01", "cam02", "cam03",
#     "cam04", "cam05", "cam06", "cam07",
#     "cam08", "cam09", "cam10", "cam11",
#     "cam12", "cam13", "cam14", "cam15",
#     "cam16", "cam17", "cam18", "cam19",
#     "cam20", "cam21", "cam22", "cam23",
# ]

# # 遍历所有帧（0000到0500）
# for frame in tqdm(range(501), desc="Processing frames"):
#     frame_str = f"{frame:04d}"  # 格式化为 0000, 0001, ..., 0500

#     # 创建当前帧的 img 和 mask 文件夹
#     frame_img_dir = os.path.join(img_output_dir, frame_str)
#     frame_mask_dir = os.path.join(mask_output_dir, frame_str)
#     os.makedirs(frame_img_dir, exist_ok=True)
#     os.makedirs(frame_mask_dir, exist_ok=True)

#     # 遍历所有相机视角文件夹
#     for camera_id in camera_ids:
#         camera_dir1 = os.path.join(input_root, "images",camera_id)
#         camera_dir2 = os.path.join(input_root, "masks",camera_id)
#         # 图片路径
#         img_path = os.path.join(camera_dir1, f"{frame:08d}.jpg")
#         # mask 路径
#         mask_path = os.path.join(camera_dir2,  f"{frame:08d}.jpg")

#         # 如果图片和 mask 存在，则复制到输出文件夹
#         if os.path.exists(img_path):
#             shutil.copy(img_path, os.path.join(frame_img_dir, f"{camera_id}.jpg"))
#         if os.path.exists(mask_path):
#             shutil.copy(mask_path, os.path.join(frame_mask_dir, f"{camera_id}.jpg"))

# # print("处理完成！")





import os
import shutil
from tqdm import tqdm

# 输入路径（output 文件夹）
input_dir = "/mnt/ssd2tB/haocheng/AnimatableGaussians/data/thuman4.0/subject01/output"

# 输出路径（parm 文件夹）
output_root = "/mnt/ssd2tC/hctang/GPS-Gaussian/new_thuman_sub01/parm"
os.makedirs(output_root, exist_ok=True)

# 遍历所有帧（0000到0500）
for frame in tqdm(range(501), desc="Creating folders"):
    frame_str = f"{frame:04d}"  # 格式化为 0000, 0001, ..., 0500

    # 创建当前帧的文件夹
    frame_dir = os.path.join(output_root, frame_str)
    os.makedirs(frame_dir, exist_ok=True)

    # 将 input_dir 的内容复制到当前帧的文件夹中
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isfile(item_path):
            shutil.copy(item_path, os.path.join(frame_dir, item))
        elif os.path.isdir(item_path):
            shutil.copytree(item_path, os.path.join(frame_dir, item))

print("处理完成！")