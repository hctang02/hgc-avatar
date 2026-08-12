# # To compute FID, first install pytorch_fid
# # pip install pytorch-fid

# import os
# import cv2 as cv
# from tqdm import tqdm
# import shutil

# from score import *

# cam_id = 0
# # ours_dir = './test_results/subject00/styleunet_gaussians3/testing__cam_%03d/batch_750000/rgb_map' % cam_id
# ours_dir = '/mnt/ssd2tB/haocheng/AnimatableGaussians/test_results/avatarrex_zzr/avatar-origin-zzr/training__cam_000/batch_300000/vanilla/rgb_map' 
# # posevocab_dir = './test_results/subject00/posevocab/testing__cam_%03d/rgb_map' % cam_id
# # tava_dir = './test_results/subject00/tava/cam_%03d' % cam_id
# # arah_dir = './test_results/subject00/arah/cam_%03d' % cam_id
# # slrf_dir = './test_results/subject00/slrf/cam_%03d' % cam_id
# # gt_dir = 'Z:/MultiviewRGB/THuman4/subject00/images/cam%02d' % cam_id
# # mask_dir = 'Z:/MultiviewRGB/THuman4/subject00/masks/cam%02d' % cam_id

# gt_dir = '/mnt/ssd2tB/haocheng/AnimatableGaussians/data/avatarrex/zzr/22010708' 
# mask_dir = '/mnt/ssd2tB/haocheng/AnimatableGaussians/data/avatarrex/zzr/22010708/mask/pha'
# frame_list = list(range(0, 499, 1))


# if __name__ == '__main__':
#     ours_metrics = Metrics()
#     # posevocab_metrics = Metrics()
#     # slrf_metrics = Metrics()
#     # arah_metrics = Metrics()
#     # tava_metrics = Metrics()

#     # shutil.rmtree('./tmp_quant')
#     os.makedirs('./tmp_quant/ours', exist_ok = True)
#     # os.makedirs('./tmp_quant/posevocab', exist_ok = True)
#     # os.makedirs('./tmp_quant/slrf', exist_ok = True)
#     # os.makedirs('./tmp_quant/arah', exist_ok = True)
#     # os.makedirs('./tmp_quant/tava', exist_ok = True)
#     os.makedirs('./tmp_quant/gt', exist_ok = True)
#     print(frame_list)
#     for frame_id in tqdm(frame_list):

#         ours_img = (cv.imread(ours_dir + '/%08d.jpg' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
#         # posevocab_img = (cv.imread(posevocab_dir + '/%08d.jpg' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
#         # slrf_img = (cv.imread(slrf_dir + '/%08d.png' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
#         # tava_img = (cv.imread(tava_dir + '/%d.jpg' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
#         # arah_img = (cv.imread(arah_dir + '/%d.jpg' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
#         gt_img = (cv.imread(gt_dir + '/%08d.jpg' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
#         mask_img = cv.imread(mask_dir + '/%08d.jpg' % frame_id, cv.IMREAD_UNCHANGED) > 128
#         gt_img[~mask_img] = 1.

#         # ours_img_cropped, posevocab_img_cropped, slrf_img_cropped, tava_img_cropped, arah_img_cropped, gt_img_cropped = \
#         ours_img_cropped, gt_img_cropped = \
#             crop_image(
#                 mask_img,
#                 512,
#                 ours_img,
#                 # posevocab_img,
#                 # slrf_img,
#                 # tava_img,
#                 # arah_img,
#                 gt_img
#             )

#         cv.imwrite('./tmp_quant/ours/%08d.png' % frame_id, (ours_img_cropped * 255).astype(np.uint8))
#         # cv.imwrite('./tmp_quant/posevocab/%08d.png' % frame_id, (posevocab_img_cropped * 255).astype(np.uint8))
#         # cv.imwrite('./tmp_quant/slrf/%08d.png' % frame_id, (slrf_img_cropped * 255).astype(np.uint8))
#         # cv.imwrite('./tmp_quant/tava/%08d.png' % frame_id, (tava_img_cropped * 255).astype(np.uint8))
#         # cv.imwrite('./tmp_quant/arah/%08d.png' % frame_id, (arah_img_cropped * 255).astype(np.uint8))
#         cv.imwrite('./tmp_quant/gt/%08d.png' % frame_id, (gt_img_cropped * 255).astype(np.uint8))

#         if ours_img is not None:
#             ours_metrics.psnr += compute_psnr(ours_img, gt_img)
#             ours_metrics.ssim += compute_ssim(ours_img, gt_img)
#             ours_metrics.lpips += compute_lpips(ours_img_cropped, gt_img_cropped)
#             ours_metrics.count += 1

#         # if posevocab_img is not None:
#         #     posevocab_metrics.psnr += compute_psnr(posevocab_img, gt_img)
#         #     posevocab_metrics.ssim += compute_ssim(posevocab_img, gt_img)
#         #     posevocab_metrics.lpips += compute_lpips(posevocab_img_cropped, gt_img_cropped)
#         #     posevocab_metrics.count += 1

#         # if slrf_img is not None:
#         #     slrf_metrics.psnr += compute_psnr(slrf_img, gt_img)
#         #     slrf_metrics.ssim += compute_ssim(slrf_img, gt_img)
#         #     slrf_metrics.lpips += compute_lpips(slrf_img_cropped, gt_img_cropped)
#         #     slrf_metrics.count += 1

#         # if arah_img is not None:
#         #     arah_metrics.psnr += compute_psnr(arah_img, gt_img)
#         #     arah_metrics.ssim += compute_ssim(arah_img, gt_img)
#         #     arah_metrics.lpips += compute_lpips(arah_img_cropped, gt_img_cropped)
#         #     arah_metrics.count += 1

#         # if tava_img is not None:
#         #     tava_metrics.psnr += compute_psnr(tava_img, gt_img)
#         #     tava_metrics.ssim += compute_ssim(tava_img, gt_img)
#         #     tava_metrics.lpips += compute_lpips(tava_img_cropped, gt_img_cropped)
#         #     tava_metrics.count += 1

#     print('Ours metrics: ', ours_metrics)
#     # print('PoseVocab metrics: ', posevocab_metrics)
#     # print('SLRF metrics: ', slrf_metrics)
#     # print('ARAH metrics: ', arah_metrics)
#     # print('TAVA metrics: ', tava_metrics)

#     print('--- Ours ---')
#     os.system('python -m pytorch_fid --device cuda {} {}'.format('./tmp_quant/ours', './tmp_quant/gt'))
#     # print('--- PoseVocab ---')
#     # os.system('python -m pytorch_fid --device cuda {} {}'.format('./tmp_quant/posevocab', './tmp_quant/gt'))
#     # print('--- SLRF ---')
#     # os.system('python -m pytorch_fid --device cuda {} {}'.format('./tmp_quant/slrf', './tmp_quant/gt'))
#     # print('--- ARAH ---')
#     # os.system('python -m pytorch_fid --device cuda {} {}'.format('./tmp_quant/arah', './tmp_quant/gt'))
#     # print('--- TAVA ---')
#     # os.system('python -m pytorch_fid --device cuda {} {}'.format('./tmp_quant/tava', './tmp_quant/gt'))


import os
import cv2 as cv
import numpy as np
import torch
import open_clip
from tqdm import tqdm
from score import *
from PIL import Image
# 初始化 CLIP 模型
clip_model, preprocess, tokenizer= open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")


def compute_clip_score(image1, image2):
    """ 计算两个图像的 CLIP 相似度 """
    # OpenCV 默认是 BGR，需要先转换为 RGB
    image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
    image2 = cv.cvtColor(image2, cv.COLOR_BGR2RGB)

    # 将图像数据从 float32 转换为 uint8 (0-255)
    image1 = (image1 * 255).astype(np.uint8)
    image2 = (image2 * 255).astype(np.uint8)

    # 转换为 PIL Image
    image1 = Image.fromarray(image1)
    image2 = Image.fromarray(image2)

    # 进行预处理
    image1 = preprocess(image1).unsqueeze(0)
    image2 = preprocess(image2).unsqueeze(0)
    
    with torch.no_grad():
        image1_features = clip_model.encode_image(image1)
        image2_features = clip_model.encode_image(image2)
        
    image1_features /= image1_features.norm(dim=-1, keepdim=True)
    image2_features /= image2_features.norm(dim=-1, keepdim=True)
    
    similarity = (image1_features @ image2_features.T).item()
    return similarity


# 文件路径设置
# ours_dir = '/mnt/ssd2tB/haocheng/AnimatableGaussians/test_results/subject02/avatar-origin-thu02/training__cam_017/batch_450000/vanilla/rgb_map' 
# ours_dir = '/mnt/ssd2tB/haocheng/GaussianAvatar/output/subject02_stage1/test_free/ours_180'
# ours_dir ='/mnt/ssd2tB/haocheng/AnimatableGaussians/test_results/subject00/avatar-face-thu00/training__cam_018---8/batch_800000/vanilla/rgb_map'
# ours_dir = '/mnt/ssd2tB/haocheng/AnimatableGaussians/test_results/subject00/avatar-face-thu00/training__cam_018---4/batch_800000/vanilla/rgb_map'
# ours_dir = '/mnt/ssd2tB/haocheng/AnimatableGaussians/test_results/subject00/avatar-origin-thu00/training__cam_018/batch_450000/vanilla/rgb_map'
# ours_dir = '/mnt/ssd2tB/haocheng/AnimatableGaussians/test_results/avatarrex_lbn2/avatar-face-lbn2-q6/training__cam_008/batch_600000/vanilla/rgb_map'
# ours_dir = '/mnt/ssd2tC/hctang/3dgs-avatar/exp/ps_female_4-none-mlp_field-ingp-shallow_mlp-default/test-pose/renders'
# ours_dir = '/mnt/ssd2tB/haocheng/AnimatableGaussians/test_results/avatarrex_zzr/avatar-face-zzr/point4_q=7_cam_7/batch_650000/vanilla/rgb_map'
# ours_dir = '/mnt/ssd2tB/haocheng/AnimatableGaussians/test_results/avatarrex_lbn1/avatar-face-lbn1/origin/batch_800000/vanilla/rgb_map'
ours_dir = '/mnt/ssd2tB/haocheng/AnimatableGaussians/test_results/subject02/avatar-face-thu02/point4_q=7_cam_17/batch_650000/vanilla/rgb_map'
# gt_dir = '/mnt/ssd2tB/haocheng/AnimatableGaussians/data/thuman4.0/subject02/images/cam17' 
# gt_dir = '/mnt/ssd2tB/haocheng/AnimatableGaussians/data/avatarrex/lbn2/22053917' 
# mask_dir = '/mnt/ssd2tB/haocheng/AnimatableGaussians/data/thuman4.0/subject02/masks/cam17'
# mask_dir = '/mnt/ssd2tB/haocheng/AnimatableGaussians/data/avatarrex/lbn2/22053917/mask/pha' 
# face_mask_dir = '/mnt/ssd2tB/haocheng/AnimatableGaussians/data/thuman4.0/subject02/images/cam17_mask'
# face_mask_dir = '/mnt/ssd2tB/haocheng/AnimatableGaussians/data/avatarrex/lbn1/22010708_mask'
# ours_dir = '/mnt/ssd2tB/haocheng/AnimatableGaussians/test_results/actor01/avatar-face-actor01---q6_img2_0/training__cam_126/batch_800000/vanilla/rgb_map' 
# gt_dir = '/mnt/ssd2tC/hctang/AnimatableGaussian/data/ActorsHQ/Actor01/Sequence1/4x/rgbs/Cam127' 
# gt_dir = '/mnt/ssd2tB/haocheng/AnimatableGaussians/data/avatarrex/zzr/22053912' 
# mask_dir = '/mnt/ssd2tB/haocheng/AnimatableGaussians/data/avatarrex/zzr/22053912/mask/pha'
# gt_dir = '/mnt/ssd2tB/haocheng/AnimatableGaussians/data/avatarrex/lbn1/22053912' 
# mask_dir = '/mnt/ssd2tB/haocheng/AnimatableGaussians/data/avatarrex/lbn1/22053912/mask/pha'
gt_dir = '/mnt/ssd2tB/haocheng/AnimatableGaussians/data/thuman4.0/subject02/images/cam17' 
mask_dir = '/mnt/ssd2tB/haocheng/AnimatableGaussians/data/thuman4.0/subject02/masks/cam17'
# face_mask_dir = '/mnt/ssd2tB/haocheng/AnimatableGaussians/data/avatarrex/lbn1/22010708_mask'
frame_list = list(range(0, 500, 2))
# frame_list = list(range(0, 20, 1))
# frame_list = list(range(0, 340, 1))s

if __name__ == '__main__':
    ours_metrics = Metrics()
    clip_score_sum = 0.0
    os.makedirs('./tmp_quant/ours', exist_ok=True)
    os.makedirs('./tmp_quant/gt', exist_ok=True)
    os.makedirs('./tmp_quant/ours_masked', exist_ok=True)
    os.makedirs('./tmp_quant/gt_masked', exist_ok=True)
    cam_id = "Cam127"  # 假设相机 ID 为 Cam127
    for frame_id in tqdm(frame_list):
        
        ours_img = (cv.imread(ours_dir + '/%08d.jpg' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        # ours_img = (cv.imread(ours_dir + 'render_c01_f/%06d.png' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        gt_img = (cv.imread(gt_dir + '/%08d.jpg' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        mask_img = cv.imread(mask_dir + '/%08d.jpg' % frame_id, cv.IMREAD_UNCHANGED) > 128
        # gt_img = (cv.imread(gt_dir + '/'+ f"Cam127_rgb{frame_id:06d}.jpg" , cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        # mask_img = cv.imread(mask_dir + '/'+ f"Cam127_mask{frame_id:06d}.png" , cv.IMREAD_UNCHANGED) > 128


        # # 构造新的文件名格式
        # gt_filename = f"{cam_id}_rgb{frame_id:06d}.jpg"
        # mask_filename = f"{cam_id}_mask{frame_id:06d}.png"

        # # 读取 GT 图像
        # gt_img = (cv.imread(gt_dir + '/' + gt_filename, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)

        # # 读取 Mask 图像
        # mask_img = cv.imread(mask_dir + '/' + mask_filename, cv.IMREAD_UNCHANGED) > 128


        # face_mask = cv.imread(face_mask_dir + '/%08d.jpg' % frame_id, cv.IMREAD_UNCHANGED) > 128
        gt_img[~mask_img] = 1.



        ours_img_cropped, gt_img_cropped = crop_image(mask_img, 512, ours_img, gt_img)
        # ours_masked_cropped, gt_masked_cropped = crop_image(face_mask, 512, ours_img, gt_img)


        cv.imwrite(f'./tmp_quant/ours/{frame_id:08d}.png', (ours_img_cropped * 255).astype(np.uint8))
        cv.imwrite(f'./tmp_quant/gt/{frame_id:08d}.png', (gt_img_cropped * 255).astype(np.uint8))
        # 
        # cv.imwrite(f'./tmp_quant/ours_masked/{frame_id:08d}.png', (ours_masked_cropped * 255).astype(np.uint8))
        # cv.imwrite(f'./tmp_quant/gt_masked/{frame_id:08d}.png', (gt_masked_cropped * 255).astype(np.uint8))

        if ours_img is not None:
            ours_metrics.psnr += compute_psnr(ours_img, gt_img)
            ours_metrics.ssim += compute_ssim(ours_img, gt_img)
            ours_metrics.lpips += compute_lpips(ours_img_cropped, gt_img_cropped)
            # ours_metrics.psnr += compute_psnr(ours_masked_cropped, gt_masked_cropped)
            # ours_metrics.ssim += compute_ssim(ours_masked_cropped, gt_masked_cropped)
            # ours_metrics.lpips += compute_lpips(ours_masked_cropped, gt_masked_cropped)
            ours_metrics.count += 1
            
            clip_score_sum += compute_clip_score(ours_img_cropped, gt_img_cropped)
            # clip_score_sum += compute_clip_score(ours_masked_cropped, gt_masked_cropped)
    
    print('Ours metrics:', ours_metrics)
    print(f'Average CLIP Score: {clip_score_sum / len(frame_list):.4f}')
    
    print('--- Ours ---')
    # os.system('python -m pytorch_fid --device cuda ./tmp_quant/ours ./tmp_quant/gt')
    os.system('python -m pytorch_fid --device cuda:0 {} {}'.format('./tmp_quant/ours', './tmp_quant/gt'))

    # print('--- Masked FID ---')
    # os.system('python -m pytorch_fid --device cuda:6 ./tmp_quant/ours_masked ./tmp_quant/gt_masked')