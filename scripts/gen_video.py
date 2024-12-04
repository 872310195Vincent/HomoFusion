import cv2
import os
from tqdm import tqdm
# from natsort import natsorted  # 用于自然排序，需要安装：pip install natsort

# 设置目录路径和参数
image_folder = '/drive/HomoFusion/outputs/2024-12-04/00-02-31/day'  # 替换为你的图片目录路径
output_video = '/drive/HomoFusion/outputs/2024-12-04/00-02-31/output_video.mp4'          # 输出视频文件名
fps = 30                                   # 设置帧率

# 获取图片文件列表并排序
images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
images = sorted(images)  # 自然排序（按文件名顺序）

# 检查图片列表是否为空
if not images:
    raise ValueError("No images found in the specified folder.")

# 获取图片的尺寸
first_image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_image_path)
height, width, layers = frame.shape
size = (width, height)

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
video = cv2.VideoWriter(output_video, fourcc, fps, size)

# 按顺序写入图片到视频
for image in tqdm(images):
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

# 释放资源
video.release()
print(f"Video saved as {output_video}")
