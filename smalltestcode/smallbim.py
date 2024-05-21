import torch
from torchvision import transforms
from PIL import Image
import os
from model import ViolenceClassifier  # 从model.py中导入模型类
from pytorch_lightning import LightningModule

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
def bim_attack(image, label, epsilon, alpha, iters):
    """Applies the BIM attack by modifying the image based on the gradient and small step size."""
    perturbed_image = image.clone().detach()  # Clone and detach to ensure it is a leaf variable
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
    if not os.path.exists(bim_directory):
        os.makedirs(bim_directory)

    # 遍历目录中的所有图片
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            image = load_image(image_path)
            label = int(filename.split('_')[0])  # 从文件名获取标签
            perturbed_image = bim_attack(image, label, epsilon, alpha, iters)  # Updated to pass label

            # 保存对抗样本
            save_path = os.path.join(bim_directory, filename)
            save_image = transforms.ToPILImage()(perturbed_image.squeeze(0))
            save_image.save(save_path)
            print(filename + ' Saved')


# 设置目录和参数
test_directory = '../smalltest/test'
bim_directory = '../smalltest/bim'
epsilon = 0.05  # 最大扰动
alpha = 0.01    # 每一步的扰动
iters = 30      # 迭代次数

# 运行代码生成和保存对抗样本
save_perturbed_images(test_directory, bim_directory, epsilon, alpha, iters)
