import cv2
import os
from tqdm import tqdm
def images_to_video(image_folder, output_video, fps=1):
    # 获取所有图片文件，按文件名排序
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png") or img.endswith(".jpg")]
    
    # 确保有图片文件
    if not images:
        raise ValueError("No images found in the folder.")
    
    # 读取第一张图片的尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # 定义视频编码器并创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式（如MP4）
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 遍历所有图片并写入视频
    for image in tqdm(images):
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"Warning: {image_path} could not be read.")
            continue
        
        video.write(frame)

    # 释放 VideoWriter 对象
    video.release()
    print(f"Video saved as {output_video}")

# 调用函数
image_folder = '/jhcnas4/syangcw/surgvu24/frames/case_145'  # 替换为你的图片文件夹路径
output_video = '/home/syangcw/SurgVU/submission/test/case_145.mp4'  # 输出视频文件名
images_to_video(image_folder, output_video, fps=1)
