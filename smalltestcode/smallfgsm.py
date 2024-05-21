import torch
from torchvision import transforms
from PIL import Image
import os
from model import ViolenceClassifier  # 从model.py中导入模型类
from pytorch_lightning import LightningModule

def fgsm_attack(image, epsilon, data_grad):
    """Applies the FGSM attack by modifying the image based on the gradient and epsilon."""
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)  # 确保扰动后的图像仍然是有效的图像数据
    return perturbed_image

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

def save_perturbed_images(directory, fgsm_directory, epsilon):
    # 确保输出文件夹存在
    if not os.path.exists(fgsm_directory):
        os.makedirs(fgsm_directory)

    # 加载模型并设置为评估模式
    model = ViolenceClassifier()
    model_path = '../model/resnet18_pretrain_test-epoch=10-val_loss=0.06.ckpt'
    model.load_from_checkpoint(model_path)
    model.eval()

    # 遍历目录中的所有图片
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            image = load_image(image_path)
            image.requires_grad = True

            # 正向传播和反向传播来获取梯度
            label = torch.tensor([int(filename.split('_')[0])])  # 从文件名获取标签
            output = model(image)
            loss = model.loss_fn(output, label)
            model.zero_grad()
            loss.backward()

            # 使用图像梯度生成对抗样本
            data_grad = image.grad.data
            perturbed_image = fgsm_attack(image, epsilon, data_grad)

            # 保存对抗样本
            save_path = os.path.join(fgsm_directory, filename)
            # 反转预处理并保存图像
            save_image = transforms.ToPILImage()(perturbed_image.squeeze(0))
            save_image.save(save_path)
            print(filename + ' Saved')

# 设置目录和epsilon
test_directory = '../smalltest/test'
fgsm_directory = '../smalltest/fgsm'
epsilon = 0.1  # 扰动系数

# 运行代码生成和保存对抗样本
save_perturbed_images(test_directory, fgsm_directory, epsilon)
