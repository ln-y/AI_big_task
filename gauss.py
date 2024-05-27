import numpy as np
from PIL import Image
import os, shutil
import argparse
import random
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, help="path of model file *.ckpt")
parser.add_argument('-eps', type=float, help="the value of `epsilon`")
parser.add_argument('-num', type=int, default=-1, help="the num of samples")
args = parser.parse_args()
print(args)

def add_gaussian_noise(image, mean=0, std=25.5):
    """
    向图像添加高斯噪声
    :param image: PIL图像对象
    :param mean: 噪声的均值
    :param std: 噪声的标准差
    :return: 带有噪声的图像对象
    """
    # 将图像转换为数组
    image_np = np.array(image)

    # 生成高斯噪声
    gaussian = np.random.normal(mean, std, image_np.shape)

    # 将噪声添加到图像
    noisy_image = image_np + gaussian

    # 确保值仍然在合理范围
    noisy_image = np.clip(noisy_image, 0, 255)

    # 转回图像格式
    noisy_image = Image.fromarray(noisy_image.astype(np.uint8))
    return noisy_image

def process_images(input_directory, output_directory, mean=0, std=25.5, num=-1):
    # 确保输出目录存在
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    # 获取输入目录中的所有文件
    files = os.listdir(input_directory)
    random.shuffle(files)
    if num > 0:
        files = files[:num]

    # 遍历输入目录中的所有文件
    for filename in tqdm(files, desc="Generating Gaussian noise dataset"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_directory, filename)
            image = Image.open(file_path).convert('RGB')
            noisy_image = add_gaussian_noise(image, mean, std)

            # 保存加噪声后的图像
            noisy_image.save(os.path.join(output_directory, filename))

# 设置输入输出目录
input_directory = 'test'
output_directory = f'gauss_eps={args.eps}'

# 执行加噪声处理
process_images(input_directory, output_directory, std=25.5, num=args.num)
