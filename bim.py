import torch
from torchvision import transforms
from PIL import Image
import os,shutil
from model import ViolenceClassifier  # 从model.py中导入模型类
from pytorch_lightning import LightningModule
from tqdm import tqdm
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('-model',type=str,help="path of model file *.ckpt")
parser.add_argument('-eps',type=float,help="the value of `epsilon`")
parser.add_argument('-num',type=int,default=-1,help="the num of samples")
args = parser.parse_args()
print(args)

device='cuda' if torch.cuda.is_available() else 'cpu'

# 加载模型并设置为评估模式
model = ViolenceClassifier()
model_path = args.model
model.load_from_checkpoint(model_path)
model.eval()
model.to(device)

def bim_attack(image, label, epsilon, alpha, iters):
    """Applies the BIM attack by modifying the image based on the gradient and small step size."""
    image=image.to(device)
    perturbed_image = image.clone().detach().to(device)  # Clone and detach to ensure it is a leaf variable
    perturbed_image.requires_grad = True

    for _ in range(iters):
        output = model(perturbed_image)
        loss = model.loss_fn(output, torch.tensor([label]).to(output.device))
        model.zero_grad()
        loss.backward()

        # Apply FGSM attack per step using the sign of the data gradient
        data_grad = perturbed_image.grad.data
        perturbed_image = perturbed_image.detach() + alpha * data_grad.sign()
        perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Ensure pixel values are in [0, 1]
        perturbed_image = torch.max(torch.min(perturbed_image, image + epsilon), image - epsilon)  # Clip perturbation
        perturbed_image.requires_grad = True  # Set requires_grad to True for next iteration

    return perturbed_image.detach()  # Detach final image to stop tracking gradients


def load_image(image_path):
    # 图像预处理步骤
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)  # 添加批处理维度
    return image


def save_perturbed_images(directory, bim_directory, epsilon, alpha, iters):
    # 确保输出文件夹存在
    if os.path.exists(bim_directory):
        shutil.rmtree(bim_directory)
    os.makedirs(bim_directory)

    files=os.listdir(directory)
    random.shuffle(files)
    if args.num>0:
        files=files[:args.num]
    # 遍历目录中的所有图片
    for filename in tqdm(files,desc="generating"):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            image = load_image(image_path)
            label = int(filename.split('_')[0])  # 从文件名获取标签
            perturbed_image = bim_attack(image, label, epsilon, alpha, iters)  # Updated to pass label

            # 保存对抗样本
            save_path = os.path.join(bim_directory, filename)
            save_image = transforms.ToPILImage()(perturbed_image.squeeze(0))
            save_image.save(save_path)
            # print(filename + ' Saved')


# 设置目录和参数
test_directory = 'test'
bim_directory = f'bim_eps={args.eps}'
epsilon = args.eps  # 最大扰动
alpha = 0.005    # 每一步的扰动
iters = 50      # 迭代次数

# 运行代码生成和保存对抗样本
save_perturbed_images(test_directory, bim_directory, epsilon, alpha, iters)
