import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

# 设置输入输出目录
input_directory = 'noise_pic'
output_directory = 'denoise_pic'

# 噪声检测阈值
noise_threshold = 0.1

def is_noisy(image, threshold=0.1):
    """判断图像是否含有噪声"""
    image_np = np.array(image)
    noise_level = np.std(image_np)
    return noise_level > threshold

def detect_noise_type(image):
    """检测噪声类型，返回 'gaussian' 或 'salt_and_pepper' 或 None"""
    image_np = np.array(image)

    # 计算图像的直方图
    hist, bins = np.histogram(image_np.flatten(), 256, [0, 256])

    # 判断是否为椒盐噪声：有较高的极值像素数量
    num_salt_pepper = hist[0] + hist[-1]
    if num_salt_pepper > 0.05 * image_np.size:
        return 'salt_and_pepper'

    # 判断是否为高斯噪声：通过图像的标准差
    noise_std = np.std(image_np)
    if noise_std > noise_threshold:
        return 'gaussian'

    return None

def remove_gaussian_noise(image):
    """高斯噪声去噪处理，使用高斯滤波"""
    image_np = np.array(image)
    # 使用高斯滤波去噪
    denoised_image = cv2.GaussianBlur(image_np, (5, 5), 1.5)
    return Image.fromarray(denoised_image)

def remove_salt_and_pepper_noise(image):
    """椒盐噪声去噪处理，使用增强的中值滤波"""
    image_np = np.array(image)
    # 使用更大的中值滤波器核尺寸
    denoised_image = cv2.medianBlur(image_np, 5)
    # 叠加多次中值滤波以增强效果
    denoised_image = cv2.medianBlur(denoised_image, 5)
    return Image.fromarray(denoised_image)

def preprocess_image(image):
    """图像预处理，包括去噪处理"""
    noise_type = detect_noise_type(image)
    if noise_type == 'gaussian':
        denoised_image = remove_gaussian_noise(image)
    elif noise_type == 'salt_and_pepper':
        denoised_image = remove_salt_and_pepper_noise(image)
    else:
        denoised_image = image

    return denoised_image

def process_images(input_directory, output_directory, noise_threshold=0.1):
    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    files = os.listdir(input_directory)

    for filename in tqdm(files, desc="Processing images"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_directory, filename)
            image = Image.open(file_path).convert('RGB')

            if is_noisy(image, threshold=noise_threshold):
                preprocessed_image = preprocess_image(image)
                # 保存去噪后的图像
                save_path = os.path.join(output_directory, filename)
                preprocessed_image.save(save_path)

if __name__ == "__main__":
    process_images(input_directory, output_directory, noise_threshold=noise_threshold)
