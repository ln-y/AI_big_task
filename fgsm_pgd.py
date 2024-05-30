import torch
from torchvision import transforms
from PIL import Image
import os, shutil
from tqdm import tqdm
import argparse
import random
from model import ViolenceClassifier  # 从model.py中导入模型类

# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, help="path of model file *.ckpt")
parser.add_argument('-eps', type=float, help="the value of `epsilon` for FGSM")
parser.add_argument('-num', type=int, default=-1, help="the num of samples to process")
args = parser.parse_args()
print(args)

# 固定的PGD参数
alpha = 0.07  # 步长
num_steps = 50  # 迭代步数

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)  # 保证图像数据在有效范围内
    return perturbed_image

def pgd_attack(model, image, epsilon, alpha, num_steps, label):
    original_image = image.data
    for _ in range(num_steps):
        image = image.detach()  # 分离张量，确保是叶子张量
        image.requires_grad = True
        output = model(image)
        model.zero_grad()
        loss = model.loss_fn(output, label)
        loss.backward()
        data_grad = image.grad.data
        sign_data_grad = data_grad.sign()
        image = image + alpha * sign_data_grad
        eta = torch.clamp(image - original_image, min=-epsilon, max=epsilon)
        image = torch.clamp(original_image + eta, 0, 1)
    return image


def load_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image)
    image = image.unsqueeze(0)
    return image

def save_perturbed_images(directory, adv_directory, epsilon):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if os.path.exists(adv_directory):
        shutil.rmtree(adv_directory)
    os.makedirs(adv_directory)

    model = ViolenceClassifier()
    model_path = args.model
    model.load_from_checkpoint(model_path)
    model.eval()
    model.to(device)

    files = os.listdir(directory)
    random.shuffle(files)
    if args.num > 0:
        files = files[:args.num]

    for filename in tqdm(files, desc="Generating adversarial examples"):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            image = load_image(image_path).to(device)
            label = torch.tensor([int(filename.split('_')[0])], device=device)
            image.requires_grad = True

            # Apply FGSM Attack
            output = model(image)
            loss = model.loss_fn(output, label)
            model.zero_grad()
            loss.backward()
            fgsm_perturbed = fgsm_attack(image, epsilon, image.grad.data)

            # Refine with PGD
            pgd_perturbed = pgd_attack(model, fgsm_perturbed, epsilon, alpha, num_steps, label)

            # Save the adversarial image
            save_path = os.path.join(adv_directory, filename)
            save_image = transforms.ToPILImage()(pgd_perturbed.squeeze(0))
            save_image.save(save_path)
            #print(f'{filename} saved at {save_path}')  # 打印调试信息

test_directory = 'test'
adv_directory = f'fgsm_pgd_eps={args.eps}'
save_perturbed_images(test_directory, adv_directory, args.eps)
