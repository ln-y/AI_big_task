import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os, shutil
from model import ViolenceClassifier  # 从model.py中导入模型类
from tqdm import tqdm
import argparse
import random
from typing import Optional
from pytorch_lightning import Trainer
from dataset import CustomDataModule

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, help="path of model file *.ckpt")
parser.add_argument('-eps', type=float, help="maximum value of `epsilon`")
parser.add_argument('-num', type=int, default=-1, help="the number of samples")
parser.add_argument('-alpha', type=float, default=0.005, help="step size for each iteration")
parser.add_argument('-iters', type=int, default=50, help="number of iterations")
args = parser.parse_args()
print(args)

def bim_attack(image: torch.Tensor, epsilon: float, alpha: float, iters: int, model: ViolenceClassifier,
               label: torch.Tensor) -> Optional[torch.Tensor]:
    """Applies the BIM attack and returns the perturbed image if the attack is successful."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = image.to(device)
    perturbed_image = image.clone().detach().to(device)  # Clone and detach to ensure it is a leaf variable
    perturbed_image.requires_grad = True

    for _ in range(iters):
        output = model(perturbed_image)
        loss = model.loss_fn(output, label)
        model.zero_grad()
        loss.backward()

        # Apply FGSM attack per step using the sign of the data gradient
        data_grad = perturbed_image.grad.data
        perturbed_image = perturbed_image.detach() + alpha * data_grad.sign()
        perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Ensure pixel values are in [0, 1]
        perturbed_image = torch.max(torch.min(perturbed_image, (image + epsilon).to(perturbed_image.dtype)),
                                    (image - epsilon).to(perturbed_image.dtype))

        pil_image = transforms.ToPILImage()(perturbed_image.squeeze(0))
        perturbed_image = transforms.ToTensor()(pil_image).unsqueeze(0).to(perturbed_image.device)  #处理浮点误差

        perturbed_image.requires_grad = True  # Set requires_grad to True for next iteration

        logits = model(perturbed_image)
        final_pred = torch.argmax(logits)

        # 如果攻击成功且预测结果发生了类别变化，返回扰动后的图像张量；否则返回 None
        if final_pred.item() != label.item() and (final_pred.item() == 0 or final_pred.item() == 1):
            return perturbed_image

    return None

def load_image(image_path) -> torch.Tensor:
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image)
    image = image.unsqueeze(0)  # 添加批处理维度
    return image

def save_perturbed_images(directory, bim_directory, epsilon, alpha, iters):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if os.path.exists(bim_directory):
        try:
            shutil.rmtree(bim_directory)
        except FileNotFoundError:
            pass
    os.makedirs(bim_directory)

    model = ViolenceClassifier()
    model.model.load_state_dict(torch.load(args.model))
    model.eval()
    model.to(device)

    files = os.listdir(directory)
    random.shuffle(files)
    if args.num > 0:
        files = files[:args.num]

    success_img = 0
    for filename in tqdm(files, desc="Generating adversarial images"):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            image = load_image(image_path).to(device)
            label = torch.tensor([int(filename.split('_')[0])]).to(device).view(-1)  # 将标签变为1D张量

            perturbed_image = bim_attack(image, epsilon, alpha, iters, model, label)
            if perturbed_image is not None:
                filename = filename.replace(".jpg", ".png")
                save_path = os.path.join(bim_directory, filename)
                save_image = transforms.ToPILImage()(perturbed_image.squeeze(0))
                save_image.save(save_path)
                success_img += 1

    # data_module.set_test_path(bim_directory)
    # trainer.test(model, datamodule=data_module)
    # torch.save(model.model.state_dict(), "model.pth")

    print(f"total success img:{success_img}")

data_module = CustomDataModule(batch_size=1, num_workers=0)  # Ensure num_workers is set to 0
test_directory = 'test'
bim_directory = f'bim_eps={args.eps}'
epsilon = args.eps  # 最大扰动
alpha = args.alpha  # 每一步的扰动
iters = args.iters  # 迭代次数
trainer = Trainer()
save_perturbed_images(test_directory, bim_directory, epsilon, alpha, iters)
