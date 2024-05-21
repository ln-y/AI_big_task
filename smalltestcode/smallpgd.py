import torch
from torchvision import transforms
from PIL import Image
import os
from model import ViolenceClassifier  # 从model.py中导入模型类

# 加载模型并设置为评估模式
model = ViolenceClassifier()
model_path = '../model/resnet18_pretrain_test-epoch=10-val_loss=0.06.ckpt'
model.load_from_checkpoint(model_path)
model.eval()

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

def pgd_attack(model, image, label, epsilon, alpha, iters):
    """Perform the PGD attack on an image."""
    original_image = image.clone().detach()  # 保存原始图像以便后续使用
    perturbed_image = original_image + 0.001 * torch.randn_like(original_image)  # 初始随机扰动

    for _ in range(iters):
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        loss = model.loss_fn(output, torch.tensor([label]).to(output.device))
        model.zero_grad()
        loss.backward()

        # 对梯度进行符号化并更新扰动图像
        data_grad = perturbed_image.grad.data
        perturbed_image = perturbed_image.detach() + alpha * data_grad.sign()
        # 使扰动后的图像保持在原始图像的 \(\epsilon\) 邻域内
        perturbation = torch.clamp(perturbed_image - original_image, -epsilon, epsilon)
        perturbed_image = torch.clamp(original_image + perturbation, 0, 1)  # 保持图像有效性

    return perturbed_image.detach()

def save_perturbed_images(directory, pgd_directory, epsilon, alpha, iters):
    # 确保输出文件夹存在
    if not os.path.exists(pgd_directory):
        os.makedirs(pgd_directory)

    # 遍历目录中的所有图片
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            image = load_image(image_path)
            label = int(filename.split('_')[0])  # 从文件名获取标签
            perturbed_image = pgd_attack(model, image, label, epsilon, alpha, iters)  # Updated to pass label

            # 保存对抗样本
            save_path = os.path.join(pgd_directory, filename)
            save_image = transforms.ToPILImage()(perturbed_image.squeeze(0))
            save_image.save(save_path)
            print(filename + ' Saved')

# 设置目录和参数
test_directory = '../smalltest/test'
pgd_directory = '../smalltest/pgd'  # 改为PGD文件夹
epsilon = 0.1  # 扰动的最大范围
alpha = 0.01    # 每步攻击的步长
iters = 40      # 迭代次数

# 运行代码生成和保存对抗样本
save_perturbed_images(test_directory, pgd_directory, epsilon, alpha, iters)