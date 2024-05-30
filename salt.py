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
parser.add_argument('-path', type=str, default="test", help="the source file data")
args = parser.parse_args()
print(args)

def add_salt_and_pepper_noise(image, noise_density=0.05):
    """
    向图像添加椒盐噪声
    :param image: PIL图像对象
    :param noise_density: 噪声密度，表示图像中将受到噪声影响的像素比例
    :return: 带有噪声的图像对象
    """
    # 将图像转换为数组
    image_np = np.array(image)

    image_size = image_np.shape[0] * image_np.shape[1]

    # 计算噪声像素的数量
    num_salt = np.ceil(noise_density * image_size * 0.5)
    num_pepper = np.ceil(noise_density * image_size * 0.5)

    # 在图像中随机选择噪声像素的位置
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image_np.shape]
    image_np[coords[0], coords[1]] = 255

    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image_np.shape]
    image_np[coords[0], coords[1]] = 0

    # 转回图像格式
    noisy_image = Image.fromarray(image_np)
    return noisy_image

def process_images(input_directory, output_directory, noise_density=0.05, num=-1):
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
    for filename in tqdm(files, desc="Generating Salt-and-Pepper noise dataset"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_directory, filename)
            image = Image.open(file_path).convert('RGB')
            noisy_image = add_salt_and_pepper_noise(image, noise_density)

            # 保存加噪声后的图像
            noisy_image.save(os.path.join(output_directory, filename))

# 设置输入输出目录
input_directory = args.path
output_directory = f'salt_eps={args.eps}'

noise_density=(args.eps*255/128)**2
# 执行加噪声处理
process_images(input_directory, output_directory, noise_density=args.eps, num=args.num)
