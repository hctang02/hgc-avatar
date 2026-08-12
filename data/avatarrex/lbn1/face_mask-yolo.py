import os
import cv2
import numpy as np
from ultralytics import YOLO

def create_face_masks(input_folder):
    # 创建存放 mask 结果的文件夹
    output_folder = f"{input_folder}_mask"
    os.makedirs(output_folder, exist_ok=True)

    # 加载 YOLOv8 人脸检测模型（确保 yolov8n-face.pt 已下载）
    model = YOLO("./yolov8n-face.pt")

    # 遍历文件夹中的所有图片
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):  # 仅处理图片
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            image = cv2.imread(input_path)
            if image is None:
                print(f"Error: Unable to load image {input_path}")
                continue
            
            height, width, _ = image.shape  # 获取图像尺寸
            mask = np.zeros((height, width), dtype=np.uint8)  # 生成全黑 mask

            results = model(image)  # 运行 YOLOv8 人脸检测

            for r in results:
                for box in r.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box[:4])  # 获取检测到的人脸坐标
                    x1, y1 = max(0, x1), max(0, y1)  # 确保坐标不越界
                    mask[y1:y2, x1:x2] = 255  # 将人脸区域填充为白色

            # 保存 mask 图片
            cv2.imwrite(output_path, mask)
            print(f"Mask saved: {output_path}")

# 示例调用
create_face_masks("22010708")  # 处理 22010708 文件夹
# mask_image = cv2.imread("22139907_mask/00000000.jpg", cv2.IMREAD_GRAYSCALE)
# print(mask_image.shape)