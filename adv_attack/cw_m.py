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
parser.add_argument('-alpha', type=float, default=1e-3, help="step size for each iteration")
parser.add_argument('-iters', type=int, default=20, help="number of iterations")
parser.add_argument('-j', type=int, default=2, help="num of multiprocess")
parser.add_argument('-test', type=str, default="../test", help="path of test")
parser.add_argument('-mode', type=str, default="black", help="black or white")
args = parser.parse_args()
print(args)

test_directory = args.test
cw_directory = f'cw_eps={args.eps}'
epsilon = args.eps  # 最大扰动
alpha = args.alpha  # 每一步的扰动
iters = args.iters  # 迭代次数
mode = args.mode


def eps_to_c(eps, p=1):
    """
    Convert FGSM epsilon to C&W's c parameter.

    :param eps: FGSM epsilon value.
    :param base_c: Base c value for reference epsilon.
    :param reference_eps: Reference epsilon value.
    :return: Approximated c value for C&W attack.
    """
    # Assume a quadratic relationship between eps and c
    return p/(eps*255)**2


c = eps_to_c(epsilon)


def cw_attack(model, image:torch.Tensor, label, c=1e-4, kappa=0, max_iter=100, learning_rate=0.01, mode='black'):
    # Set device
    global perturbed_image
    device = next(model.parameters()).device
    image = image.to(device)
    label = torch.tensor([label]).to(device)

    # Define box constraints [0, 1]
    box_min = torch.zeros_like(image).to(device)
    box_max = torch.ones_like(image).to(device)

    # Initialize perturbation
    w = torch.arctanh((image-0.5)*2)
    w.requires_grad = True

    optimizer = torch.optim.Adam([w], lr=learning_rate)

    for i in range(max_iter):
        
        perturbed_image = torch.tanh(w) * 0.5 + 0.5  # Scale to [0, 1]
        # perturbed_image = image + perturbed_image * (box_max - box_min)
        output = model(perturbed_image)
        # input(f"{torch.mean(perturbed_image)=}\n{torch.mean(image)=}")
        real = output[0][label]
        other = output[0][(label.item()+1)%2]
        # input(f"{output=}\n{real=}\n{other=}")

        loss1 = torch.relu(real - other + kappa)
        loss2 = torch.mean((perturbed_image - image) ** 2)
        loss = loss1 + c * loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        perturbed_image = torch.tanh(w) * 0.5 + 0.5
        perturbed_image = ((perturbed_image * 255).to(torch.uint8) / 255).to(torch.float)
        # 白盒攻击，每次迭代均检测
        if mode == "white":
            logits = model(perturbed_image)
            final_pred = torch.argmax(logits)

            # 如果攻击成功且预测结果发生了类别变化，返回扰动后的图像张量；否则返回 None
            if final_pred.item() != label.item() and (final_pred.item() == 0 or final_pred.item() == 1):
                return perturbed_image

    # 黑盒攻击：最后检测
    logits = model(perturbed_image)
    final_pred = torch.argmax(logits)
    if final_pred.item() != label.item() and (final_pred.item() == 0 or final_pred.item() == 1):
        return perturbed_image
    return None

def load_image(image_path, device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    image = torch.from_numpy(image).permute(2, 0, 1).to(device).contiguous()  # 调整维度顺序
    image = (image / 255).to(torch.float)
    image = image.unsqueeze(0)  # 添加批处理维度
    return image


def save_perturbed_images(directory, cw_directory):
    """This function manages the parallel generation and saving of adversarial images."""
    if os.path.exists(cw_directory):
        try:
            shutil.rmtree(cw_directory)
        except FileNotFoundError:
            pass
    os.makedirs(cw_directory)

    files = os.listdir(directory)
    random.shuffle(files)
    if args.num > 0:
        files = files[:args.num]

    arr = mp.Array('i', [0] * args.j)  # Multiprocessing array to keep track of progress
    que = mp.Queue()  # Queue to store the count of successful images
    p_lst = []
    per_task_len = len(files) / args.j

    for i in range(args.j):
        p = mp.Process(target=task, args=(
        files[int(per_task_len * i):int(per_task_len * (i + 1))], directory, args.model, arr, i, que))
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

    print(f"success:{suc_num}\nsuccess ratio:{suc_num / len(files)}")


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
            image = load_image(image_path, device)
            label = torch.tensor([int(filename.split('_')[0])]).to(device).view(-1)  # 将标签变为1D张量

            perturbed_image = cw_attack(model, image, label, c=c, kappa=0, max_iter=iters, learning_rate=alpha,
                                        mode=mode)
            if perturbed_image is not None:
                filename = filename.replace(".jpg", ".png")
                save_path = os.path.join(cw_directory, filename)
                save_image = transforms.ToPILImage()(perturbed_image.squeeze(0))
                save_image.save(save_path)
                success_img += 1
        arr[id] += 1
    queue.put(success_img)


if __name__ == '__main__':
    save_perturbed_images(test_directory, cw_directory)
