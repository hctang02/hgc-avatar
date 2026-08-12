import os
import shutil
import subprocess
from PIL import Image

# === 参数区 ===
root_dir = '/mnt/ssd2tB/haocheng/AnimatableGaussians/compress_part/data'
data_name = 'output-thu00-test'
num_bins = 100000
batch_name = 'batch_650000'
q_index = 5

# === 路径区 ===
src_img_dir = os.path.join(root_dir, 'before_compress', data_name, 'pose_map_img')
encode_frame_dir = os.path.join(root_dir, 'test', data_name, 'pose_map', 'encode_frame')
decode_root = os.path.join(root_dir, 'test', data_name, 'pose_map', 'decode_frame', 'encode_frame')
bin_src = os.path.join(root_dir, 'test', data_name, 'pose_map', 'bin', 'encode_frame', '0')
pose_map_img_dst = os.path.join(root_dir, 'after_compress', data_name, 'pose_map_img')
bin_dst = os.path.join(root_dir, 'bin', data_name, 'pose_map')

os.makedirs(encode_frame_dir, exist_ok=True)
os.makedirs(pose_map_img_dst, exist_ok=True)
os.makedirs(bin_dst, exist_ok=True)

# === 步骤一：复制图片 & 转换为 png ===
print("步骤一：复制和转换图片格式...")
for filename in os.listdir(src_img_dir):
    if filename.startswith("pose_map_") and filename.endswith(".png"):
        try:
            idx = int(filename.replace("pose_map_", "").replace(".png", ""))
            src_path = os.path.join(src_img_dir, filename)
            dst_path = os.path.join(encode_frame_dir, f"im{idx+1:05d}.png")  # 注意+1
            image = Image.open(src_path)
            image.save(dst_path)
        except ValueError:
            print(f"跳过非法文件名：{filename}")
print("✅ 图片处理完成")

# === 步骤二：调用 test_video.py ===
print("步骤二：压缩中...")
codec_dir = '/mnt/ssd2tB/haocheng/AnimatableGaussians/compress_part/codec/DCVC-DC'
test_video_path = os.path.join(codec_dir, 'test_video.py')

# 自动帧数
frame_count = len([f for f in os.listdir(encode_frame_dir) if f.endswith(".png")])
print(f"自动检测帧数为 {frame_count}，将传给 test_video.py")

command = [
    "python", test_video_path,
    "--i_frame_model_path", "./checkpoints/cvpr2023_image_psnr.pth.tar",
    "--p_frame_model_path", "./checkpoints/cvpr2023_video_psnr.pth.tar",
    "--rate_num", "2",
    "--test_config", "./dataset_config_rgb.json",
    "--yuv420", "0",
    "--cuda", "1",
    "--worker", "1",
    "--write_stream", "1",
    "--verbose", "1",
    "--stream_path", os.path.join(root_dir, "test", data_name, "pose_map", "bin"),
    "--output_path", "output.json",
    "--force_intra_period", "32",
    "--force_frame_num", str(frame_count),
    "--save_decoded_frame", "1",
    "--decoded_frame_path", os.path.join(root_dir, "test", data_name, "pose_map", "decode_frame"),
    "--root_path", os.path.join(root_dir, "test", data_name)
]
result = subprocess.run(command, cwd=codec_dir)
if result.returncode == 0:
    print("✅ 压缩完成")
else:
    raise RuntimeError(f"❌ test_video.py 执行失败，返回码：{result.returncode}")

# === 步骤三：拷贝 decode_frame 的图像 + bin 文件 ===
print("步骤三：复制解码后的图像和 bin 文件...")
subfolders = [f for f in os.listdir(decode_root) if f.startswith('0') and os.path.isdir(os.path.join(decode_root, f))]
if not subfolders:
    raise FileNotFoundError("未找到以 0 开头的 decode_frame 子文件夹")
decode_img_dir = os.path.join(decode_root, subfolders[0])

for filename in os.listdir(decode_img_dir):
    if filename.startswith("im") and filename.endswith(".png"):
        try:
            idx = int(filename.replace("im", "").replace(".png", ""))
            src = os.path.join(decode_img_dir, filename)
            dst = os.path.join(pose_map_img_dst, f"pose_map_{idx - 1}.jpg")  # 注意-1
            img = Image.open(src)
            img.save(dst, format="JPEG")
        except Exception as e:
            print(f"跳过 {filename}: {e}")

for fname in os.listdir(bin_src):
    src_path = os.path.join(bin_src, fname)
    dst_path = os.path.join(bin_dst, fname)
    if os.path.isfile(src_path):
        shutil.copyfile(src_path, dst_path)
print("✅ 文件复制完成")

# === 步骤四：调用 arithmetic_encoder.py 处理 smpl_params ===
print("步骤四：开始压缩 SMPL 参数...")
arith_script_path = '/mnt/ssd2tB/haocheng/AnimatableGaussians/compress_part/codec/quantization/arithmetic_encoder.py'
arith_command = [
    'python', arith_script_path,
    '--root_dir', root_dir,
    '--data_name', data_name,
    '--num_bins', str(num_bins)
]
result = subprocess.run(arith_command)
if result.returncode == 0:
    print("✅ SMPL 参数压缩完成")
else:
    raise RuntimeError(f"❌ arithmetic_encoder.py 执行失败，返回码：{result.returncode}")

# === 步骤五：调用 core_coder.py 压缩 avatar_net ===
print("步骤五：开始压缩 Avatar 网络参数...")
core_coder_path = '/mnt/ssd2tB/haocheng/AnimatableGaussians/compress_part/codec/quantization/core_coder.py'
core_coder_command = [
    'python', core_coder_path,
    '--root_dir', root_dir,
    '--data_name', data_name,
    '--q_index', str(q_index),
    '--batch_name', batch_name
]
result = subprocess.run(core_coder_command)
if result.returncode == 0:
    print("✅ Avatar 参数压缩完成")
else:
    raise RuntimeError(f"❌ core_coder.py 执行失败，返回码：{result.returncode}")

print("🎉 所有步骤已完成！")
