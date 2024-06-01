import torch
import numpy as np
import multiprocessing as mp
from torchvision import transforms
from PIL import Image
import os, shutil
from model import ViolenceClassifier  # 从model.py中导入模型类
from tqdm import tqdm
import argparse
import random
from typing import Optional, List
import time

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, help="path of model file *.ckpt")
parser.add_argument('-eps', type=float, help="maximum value of `epsilon`")
parser.add_argument('-num', type=int, default=-1, help="the number of samples")
parser.add_argument('-alpha', type=float, default=0.005, help="step size for each iteration")
parser.add_argument('-iters', type=int, default=50, help="number of iterations")
parser.add_argument('-j',type=int,default=2,help="num of multiprocess")
parser.add_argument('-test',type=str,default="../test",help="path of test")
args = parser.parse_args()
print(args)

test_directory = args.test
pgd_directory = f'pgd_eps={args.eps}'
epsilon = args.eps  # 最大扰动
alpha = args.alpha  # 每一步的扰动
iters = args.iters  # 迭代次数

def pgd_attack(image: torch.Tensor, epsilon: float, alpha: float, iters: int, model: ViolenceClassifier,
               label: torch.Tensor) -> Optional[torch.Tensor]:
    """Applies the PGD attack and returns the perturbed image if the attack is successful."""
    device = image.device
    perturbed_image = image.clone().detach().to(device)  # Clone and detach to ensure it is a leaf variable
    perturbed_image.requires_grad = True

    for _ in range(iters):
        output = model(perturbed_image)
        loss = model.loss_fn(output, label)
        model.zero_grad()
        loss.backward()

        # Apply FGSM attack per step using the sign of the data gradient
        data_grad = perturbed_image.grad.data
        perturbed_image = perturbed_image + alpha * data_grad.sign()
        perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Ensure pixel values are in [0, 1]
        perturbed_image = torch.max(torch.min(perturbed_image, (image + epsilon).to(perturbed_image.dtype)),
                                    (image - epsilon).to(perturbed_image.dtype))

        # faster method
        perturbed_image = ((perturbed_image * 255).to(torch.uint8) / 255).to(torch.float)

        # pil_image = transforms.ToPILImage()(perturbed_image.squeeze(0))
        # perturbed_image = transforms.ToTensor()(pil_image).unsqueeze(0).to(perturbed_image.device)  #处理浮点误差

        perturbed_image.requires_grad = True  # Set requires_grad to True for next iteration

        logits = model(perturbed_image)
        final_pred = torch.argmax(logits)

        # 如果攻击成功且预测结果发生了类别变化，返回扰动后的图像张量；否则返回 None
        if final_pred.item() != label.item() and (final_pred.item() == 0 or final_pred.item() == 1):
            return perturbed_image

    return None


def load_image(image_path,device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    image = torch.from_numpy(image).permute(2, 0, 1).to(device).contiguous()  # 调整维度顺序
    image = (image/255).to(torch.float)
    image = image.unsqueeze(0)  # 添加批处理维度
    return image

def save_perturbed_images(directory, pgd_directory):
    """This function manages the parallel generation and saving of adversarial images."""
    if os.path.exists(pgd_directory):
        try:
            shutil.rmtree(pgd_directory)
        except FileNotFoundError:
            pass
    os.makedirs(pgd_directory)

    files = os.listdir(directory)
    random.shuffle(files)
    if args.num > 0:
        files = files[:args.num]

    arr = mp.Array('i', [0] * args.j)  # Multiprocessing array to keep track of progress
    que = mp.Queue()  # Queue to store the count of successful images
    p_lst = []
    per_task_len = len(files) / args.j

    for i in range(args.j):
        p = mp.Process(target=task, args=(files[int(per_task_len * i):int(per_task_len * (i + 1))], directory, args.model, arr, i, que))
        p_lst.append(p)
        p.start()

    pbar = tqdm(total=len(files), desc="generating")
    last_num = 0
    now_num = 0
    while now_num < len(files):
        last_num = now_num
        now_num = sum(arr)
        pbar.update(now_num - last_num)
        time.sleep(0.5)

    pbar.close()
    p.join()
    suc_num = 0
    while not que.empty():
        suc_num += que.get()

    print(f"success:{suc_num}\nsuccess ratio:{suc_num/len(files)}")

def task(files: List[str], directory: str, model_path: str, arr, id: int, queue):
    """Worker function for each process to generate adversarial images."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ViolenceClassifier()
    model.model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    success_img = 0
    for filename in files:
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            image = load_image(image_path,device)
            label = torch.tensor([int(filename.split('_')[0])]).to(device).view(-1)  # 将标签变为1D张量

            perturbed_image = pgd_attack(image, epsilon, alpha, iters, model, label)
            if perturbed_image is not None:
                filename = filename.replace(".jpg", ".png")
                save_path = os.path.join(pgd_directory, filename)
                save_image = transforms.ToPILImage()(perturbed_image.squeeze(0))
                save_image.save(save_path)
                success_img += 1
        arr[id] += 1
    queue.put(success_img)

if __name__ == '__main__':
    save_perturbed_images(test_directory, pgd_directory)
