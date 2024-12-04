import os.path
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import pytorch_lightning as pl
import hydra
from PIL import Image, ImageDraw, ImageFont
import time

from homo_transformer.common import setup_config, setup_network, setup_data_module
from homo_transformer.data.apolloscape_dataset import encode, get_palettedata
from homo_transformer.common import load_backbone

water_hazart = False #True

if water_hazart:
    CHECKPOINT_PATH = Path.cwd() / 'data' / 'water_hazart' / 'model.ckpt'
    Test_name_prefix = 'on_' 
    Test_name_suffix = '.png'
    save_dir = 'on_road'
    ori_size = (360, 1280) 
    pad_size = (1, 360, 1280)
else:
    CHECKPOINT_PATH = Path.cwd() / 'data' / 'apolloscape' / 'model.ckpt'
    Test_name_prefix = 'day_'
    Test_name_suffix = '.png'
    save_dir = 'day' 
    ori_size = (1084, 3384) 
    pad_size = (1, 1626, 3384)


def setup(cfg):

    cfg.loader.batch_size = 1

    if 'mixed_precision' not in cfg:
        cfg.mixed_precision = False

    if 'device' not in cfg:
        cfg.device = 'cuda'


# 添加一个函数，用来等比例压缩图像
def resize_image(image, target_width, target_height):
    # 获取原始尺寸
    width, height = image.size
    aspect_ratio = width / height

    # 计算缩放后的尺寸，保持宽高比
    if width > height:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    # 如果缩放后的尺寸超过了目标尺寸，就调整为目标尺寸
    if new_width > target_width or new_height > target_height:
        if new_width > new_height:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)

    # 调整图像大小
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image

@hydra.main(config_path=Path.cwd() / 'config', config_name='config.yaml')
def main(cfg):
    setup_config(cfg, setup)

    pl.seed_everything(2022, workers=True)

    data = setup_data_module(cfg)
    loader = data.val_dataloader(shuffle=False)

    device = torch.device(cfg.device)

    network = load_backbone(CHECKPOINT_PATH)
    network = network.to(device)
    network.eval()

    resize = transforms.Resize(ori_size, interpolation=transforms.InterpolationMode.NEAREST)

    i = 0
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 设置视频编码器为 'mp4v'（H.264 编码）
    # video_writer = cv2.VideoWriter(save_dir + '/output_video.mp4', fourcc, 30, (ori_size[1] * 3, ori_size[0]))  # 输出 MP4 视频文件
    # 将 'mp4v' 更换为 'MJPG' 编码
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 设置视频编码
    # video_writer = cv2.VideoWriter('/drive/HomoFusion/outputs/output_video.avi', fourcc, 30, (ori_size[1] * 3, ori_size[0]))  # 输出 MJPG 编码的视频文件

    # print(save_dir + '/output_video.mp4')
    
    with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
        with torch.no_grad():
            for batch in tqdm(loader):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                # print(batch.keys())
                torch.cuda.synchronize()
                start_time = time.perf_counter()

                # 使用 origin_image 和 origin_mask（从 DataLoader 中读取）
                original_pil = Image.open(batch['frame_path'][0], 'r').convert('RGB')
                gt_pil = Image.open(batch['mask_path'][0], 'r')

                # 获取预测结果，并进行后处理
                pred = network(batch)
                if water_hazart:
                    img = torch.sigmoid(pred['mask']) > 0.4
                    img = img.squeeze(1)  # 去掉通道维度
                else:
                    img = encode(pred['mask'])

                # Resize to original size
                img = resize(img)
                img = torch.cat((torch.zeros(pad_size).to(img), img), dim=-2)  # padding操作

                # 将预测结果转换为PIL图像
                if water_hazart:
                    pred_pil = F.to_pil_image(img.squeeze(0).float(), 'L')
                else:
                    pred_pil = F.to_pil_image(img.squeeze(0), 'L')
                    pred_pil = pred_pil.convert('P')  # 使用调色板
                    pred_pil.putpalette(get_palettedata())

                # 添加文本标签
                def add_label_to_image(pil_img, label):
                    # 创建一个可以在图像上绘制的对象
                    draw = ImageDraw.Draw(pil_img)
                    
                    # 加载字体（默认字体）
                    font = ImageFont.load_default()
                    
                    # 计算文本的宽度和高度
                    text_bbox = draw.textbbox((0, 0), label, font=font)  # 获取文本的边界框
                    text_width = text_bbox[2] - text_bbox[0]  # 计算宽度
                    text_height = text_bbox[3] - text_bbox[1]  # 计算高度
                    
                    # 在图像上绘制文本，居中对齐
                    draw.text(((pil_img.width - text_width) / 2, 10), label, font=font, fill="white")
                    
                    return pil_img

                original_pil = add_label_to_image(original_pil, "Original")
                gt_pil = add_label_to_image(gt_pil, "Ground Truth")
                pred_pil = add_label_to_image(pred_pil, "Predicted")

                # 等比例压缩每一张图像
                max_width = 640  # 目标宽度
                max_height = 480  # 目标高度

                original_pil = resize_image(original_pil, max_width, max_height)
                gt_pil = resize_image(gt_pil, max_width, max_height)
                pred_pil = resize_image(pred_pil, max_width, max_height)

                # 合并三张图片（水平排列）
                combined_width = original_pil.width + gt_pil.width + pred_pil.width
                combined_height = original_pil.height  # 高度保持一致
                combined_image = Image.new('RGB', (combined_width, combined_height))

                # 粘贴图片
                combined_image.paste(original_pil, (0, 0))
                combined_image.paste(gt_pil.convert('RGB'), (original_pil.width, 0))
                combined_image.paste(pred_pil.convert('RGB'), (original_pil.width + gt_pil.width, 0))

                # 将合并的图像转换为numpy数组，并写入视频
                combined_frame = np.array(combined_image)
                combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)  # 转换为BGR格式以便OpenCV使用
                # video_writer.write(combined_frame)

                # 保存合成后的图片
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                if 'name' in batch.keys():
                    image_name = '%s%09d%s' % (Test_name_prefix, batch['name'].item(), Test_name_suffix)
                else:
                    image_name = '%s%09d%s' % (Test_name_prefix, i, Test_name_suffix)

                combined_image.save(os.path.join(save_dir, image_name))

                i += 1
                # if i >= 20:
                #     break

    # video_writer.release()  # 释放视频写入器
# def main(cfg):
#     setup_config(cfg, setup)

#     pl.seed_everything(2022, workers=True)

#     data = setup_data_module(cfg)
#     loader = data.val_dataloader(shuffle=False)

#     device = torch.device(cfg.device)

#     network = load_backbone(CHECKPOINT_PATH)
#     network = network.to(device)
#     network.eval()

#     resize = transforms.Resize(ori_size, interpolation=transforms.InterpolationMode.NEAREST)

#     i = 0
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 设置视频编码器为 'mp4v'（H.264 编码）
#     video_writer = cv2.VideoWriter(save_dir + '/output_video.mp4', fourcc, 30, (ori_size[1] * 3, ori_size[0]))  # 输出 MP4 视频文件
#     print(save_dir + '/output_video.mp4')
#     with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
#         with torch.no_grad():
#             for batch in tqdm(loader):
#                 batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
#                 print(batch.keys())
#                 torch.cuda.synchronize()
#                 start_time = time.perf_counter()

#                 # 使用 origin_image 和 origin_mask（从 DataLoader 中读取）
#                 original_pil = Image.open(batch['frame_path'][0], 'r').convert('RGB')
#                 gt_pil = Image.open(batch['mask_path'][0], 'r')

#                 # 获取预测结果，并进行后处理
#                 pred = network(batch)
#                 if water_hazart:
#                     img = torch.sigmoid(pred['mask']) > 0.4
#                     img = img.squeeze(1)  # 去掉通道维度
#                 else:
#                     img = encode(pred['mask'])

#                 # Resize to original size
#                 img = resize(img)
#                 img = torch.cat((torch.zeros(pad_size).to(img), img), dim=-2)  # padding操作

#                 # 将预测结果转换为PIL图像
#                 if water_hazart:
#                     pred_pil = F.to_pil_image(img.squeeze(0).float(), 'L')
#                 else:
#                     pred_pil = F.to_pil_image(img.squeeze(0), 'L')
#                     pred_pil = pred_pil.convert('P')  # 使用调色板
#                     pred_pil.putpalette(get_palettedata())

#                 # 添加文本标签
#                 def add_label_to_image(pil_img, label):
#                     # 创建一个可以在图像上绘制的对象
#                     draw = ImageDraw.Draw(pil_img)
                    
#                     # 加载字体（默认字体）
#                     font = ImageFont.load_default()
                    
#                     # 计算文本的宽度和高度
#                     text_bbox = draw.textbbox((0, 0), label, font=font)  # 获取文本的边界框
#                     text_width = text_bbox[2] - text_bbox[0]  # 计算宽度
#                     text_height = text_bbox[3] - text_bbox[1]  # 计算高度
                    
#                     # 在图像上绘制文本，居中对齐
#                     draw.text(((pil_img.width - text_width) / 2, 10), label, font=font, fill="white")
                    
#                     return pil_img

#                 original_pil = add_label_to_image(original_pil, "Original")
#                 gt_pil = add_label_to_image(gt_pil, "Ground Truth")
#                 pred_pil = add_label_to_image(pred_pil, "Predicted")

#                 # 打印图片形状（宽度和高度）
#                 # print("Original Image Shape: ", original_pil.size)
#                 # print("Ground Truth Shape: ", gt_pil.size)
#                 # print("Predicted Image Shape: ", pred_pil.size)

#                 # 合并三张图片（水平排列）
#                 combined_width = original_pil.width + gt_pil.width + pred_pil.width
#                 combined_height = original_pil.height  # 高度保持一致
#                 combined_image = Image.new('RGB', (combined_width, combined_height))

#                 # 粘贴图片
#                 combined_image.paste(original_pil, (0, 0))
#                 combined_image.paste(gt_pil.convert('RGB'), (original_pil.width, 0))
#                 combined_image.paste(pred_pil.convert('RGB'), (original_pil.width + gt_pil.width, 0))

#                 # 将合并的图像转换为numpy数组，并写入视频
#                 combined_frame = np.array(combined_image)
#                 combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)  # 转换为BGR格式以便OpenCV使用
#                 video_writer.write(combined_frame)

#                 # 保存合成后的图片
#                 if not os.path.exists(save_dir):
#                     os.mkdir(save_dir)

#                 if 'name' in batch.keys():
#                     image_name = '%s%09d%s' % (Test_name_prefix, batch['name'].item(), Test_name_suffix)
#                 else:
#                     image_name = '%s%09d%s' % (Test_name_prefix, i, Test_name_suffix)

#                 combined_image.save(os.path.join(save_dir, image_name))

#                 i += 1
#                 if i >= 20:
#                     break

#     video_writer.release()  # 释放视频写入器





if __name__ == '__main__':
    main()
