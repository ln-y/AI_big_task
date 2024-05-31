import torch
import torch.optim as optim
import torch.nn.functional as F
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
parser.add_argument('-j', type=int, default=2, help="num of multiprocess")
args = parser.parse_args()
print(args)

test_directory = 'test'
cw_directory = f'cw_eps={args.eps}'
epsilon = args.eps * 100 # 最大扰动
alpha = args.alpha  # 每一步的扰动
iters = args.iters  # 迭代次数


def cw_attack(image: torch.Tensor, epsilon: float, alpha: float, iters: int, model: ViolenceClassifier,
              label: torch.Tensor) -> Optional[torch.Tensor]:
    """Applies the C&W attack and returns the perturbed image if the attack is successful."""
    device = image.device
    image = image.to(device)
    label = label.to(device)

    # 初始化优化变量
    perturbed_image = image.clone().detach().requires_grad_(True).to(device)
    optimizer = optim.Adam([perturbed_image], lr=alpha)

    for _ in range(iters):
        optimizer.zero_grad()

        # 计算模型输出和损失
        output = model(perturbed_image)
        loss = F.cross_entropy(output, label) + F.mse_loss(perturbed_image, image)
        loss.backward()
        optimizer.step()

        # 约束扰动
        perturbed_image.data = torch.clamp(perturbed_image.data, 0, 1)
        perturbed_image.data = torch.max(torch.min(perturbed_image.data, image + epsilon), image - epsilon)

        # 更快方法来处理浮点误差
        perturbed_image.data = ((perturbed_image.data * 255).to(torch.uint8) / 255).to(torch.float)

        # 预测扰动后的图像
        logits = model(perturbed_image)
        final_pred = torch.argmax(logits)

        # 如果攻击成功且预测结果发生了类别变化，返回扰动后的图像张量；否则返回 None
        if final_pred.item() != label.item() and (final_pred.item() == 0 or final_pred.item() == 1):
            return perturbed_image.detach()

    return None


def load_image(image_path) -> torch.Tensor:
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image)
    image = image.unsqueeze(0)  # 添加批处理维度
    return image


def save_perturbed_images(directory, cw_directory):
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

    arr = mp.Array('i', [0] * args.j)
    que = mp.Queue()
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ViolenceClassifier()
    model.model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    success_img = 0
    for filename in files:
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            image = load_image(image_path).to(device)
            label = torch.tensor([int(filename.split('_')[0])]).to(device).view(-1)  # 将标签变为1D张量

            perturbed_image = cw_attack(image, epsilon, alpha, iters, model, label)
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
