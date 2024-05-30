import torch
from torchvision import transforms
from PIL import Image
import os, shutil
from model import ViolenceClassifier  # 从 model.py 中导入模型类
from tqdm import tqdm
import argparse
import random
from typing import Optional

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, help="path of model file *.ckpt")
parser.add_argument('-eps', type=float, help="maximum value of `epsilon`")
parser.add_argument('-num', type=int, default=-1, help="the number of samples")
parser.add_argument('-step', type=float, default=0.01, help="step size for increasing epsilon")
args = parser.parse_args()
print(args)

def fgsm_attack(image: torch.Tensor, epsilon: float, model: ViolenceClassifier, label: torch.Tensor) -> Optional[torch.Tensor]:
    """Applies the FGSM attack and returns the perturbed image if the attack is successful."""
    image.requires_grad = True
    output = model(image)

    loss = model.loss_fn(output, label)
    model.zero_grad()
    loss.backward()

    data_grad = image.grad.data
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # 重新预测扰动后的图像
    output = model(perturbed_image)
    final_pred = output.max(1, keepdim=True)[1]

    # 如果攻击成功且预测结果发生了类别变化，返回扰动后的图像张量；否则返回 None
    if final_pred.item() != label.item() and (final_pred.item() == 0 or final_pred.item() == 1):
        return perturbed_image
    else:
        return None

def load_image(image_path) -> torch.Tensor:
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image)
    image = image.unsqueeze(0)  # 添加批处理维度
    return image

def save_perturbed_images(directory, fgsm_directory, max_epsilon, step):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(fgsm_directory):
        os.makedirs(fgsm_directory)

    model = ViolenceClassifier()
    model_path = args.model
    model.load_from_checkpoint(model_path)
    model.eval()
    model.to(device)

    files = os.listdir(directory)
    random.shuffle(files)
    if args.num > 0:
        files = files[:args.num]

    for filename in tqdm(files, desc="Generating adversarial images"):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            image = load_image(image_path).to(device)
            label = torch.tensor([int(filename.split('_')[0])]).to(device).view(-1)  # 将标签变为1D张量

            epsilon = step  # 从较小的扰动开始
            while epsilon <= max_epsilon:
                perturbed_image = fgsm_attack(image, epsilon, model, label)
                if perturbed_image is not None:
                    save_path = os.path.join(fgsm_directory, filename)
                    save_image = transforms.ToPILImage()(perturbed_image.squeeze(0))
                    save_image.save(save_path)
                    print(f"Saved perturbed image {filename} with epsilon {epsilon}")
                    break
                epsilon += step

test_directory = 'test'
fgsm_directory = f'new_fgsm_eps={args.eps}'
max_epsilon = args.eps
step_size = args.step

save_perturbed_images(test_directory, fgsm_directory, max_epsilon, step_size)
