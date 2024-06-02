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
from typing import Optional, List, Tuple
import time


def fgsm_attack(image: torch.Tensor, epsilon: float, alpha: float, iters: int, model: ViolenceClassifier,
                label: torch.Tensor) -> Optional[torch.Tensor]:
    """Applies the FGSM attack and returns the perturbed image if the attack is successful."""
    device = image.device
    perturbed_image = image.clone().detach().to(device)  # Clone and detach to ensure it is a leaf variable
    perturbed_image.requires_grad = True

    # 只进行一次FGSM攻击
    output = model(perturbed_image)
    loss = model.loss_fn(output, label)
    model.zero_grad()
    loss.backward()

    # Apply FGSM attack using the sign of the data gradient
    data_grad = perturbed_image.grad.data
    perturbed_image = perturbed_image.detach() + epsilon * data_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Ensure pixel values are in [0, 1]

    # faster method
    perturbed_image = ((perturbed_image * 255).to(torch.uint8) / 255).to(torch.float)

    perturbed_image.requires_grad = True  # Set requires_grad to True for next iteration

    logits = model(perturbed_image)
    final_pred = torch.argmax(logits)

    # 如果攻击成功且预测结果发生了类别变化，返回扰动后的图像张量；否则返回 None
    if final_pred.item() != label.item() and (final_pred.item() == 0 or final_pred.item() == 1):
        return perturbed_image

    return None

def pgd_attack(image: torch.Tensor, epsilon: float, alpha: float, iters: int, model: ViolenceClassifier,
               label: torch.Tensor) -> Optional[torch.Tensor]:
    """Applies the PGD attack and returns the perturbed image if the attack is successful."""
    device = image.device
    perturbed_image = image.clone().detach().to(device)  # Clone and detach to ensure it is a leaf variable

    # Add initial random noise
    noise = torch.empty_like(perturbed_image).uniform_(-alpha, alpha).to(device)
    perturbed_image = perturbed_image + noise
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    for _ in range(iters):
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        loss = model.loss_fn(output, label)
        model.zero_grad()
        loss.backward()

        # Apply FGSM attack per step using the sign of the data gradient
        data_grad = perturbed_image.grad.data
        perturbed_image = perturbed_image.detach() + alpha * data_grad.sign()
        perturbed_image = torch.max(torch.min(perturbed_image, image + epsilon), image - epsilon)
        perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Ensure pixel values are in [0, 1]

        # faster method
        perturbed_image = ((perturbed_image * 255).to(torch.uint8) / 255).to(torch.float)

    logits = model(perturbed_image)
    final_pred = torch.argmax(logits)

    # 如果攻击成功且预测结果发生了类别变化，返回扰动后的图像张量；否则返回 None
    if final_pred.item() != label.item() and (final_pred.item() == 0 or final_pred.item() == 1):
        return perturbed_image

    return None

def bim_attack(image: torch.Tensor, epsilon: float, alpha: float, iters: int, model: ViolenceClassifier,
               label: torch.Tensor) -> Optional[torch.Tensor]:
    """Applies the BIM attack and returns the perturbed image if the attack is successful."""
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
        perturbed_image = perturbed_image.detach() + alpha * data_grad.sign()
        perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Ensure pixel values are in [0, 1]
        perturbed_image = torch.max(torch.min(perturbed_image, (image + epsilon).to(perturbed_image.dtype)),
                                    (image - epsilon).to(perturbed_image.dtype))

        # faster method
        perturbed_image = ((perturbed_image*255).to(torch.uint8)/255).to(torch.float)

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


def task(rmodel:ViolenceClassifier, taskque:"mp.Queue[Tuple[List[str],int]]",
          src_dir, dst_dir, arr, id:int, eps, alpha,iters):
    device = rmodel.device
    if id>0:
        model = ViolenceClassifier()
        model.model.load_state_dict(rmodel.model.state_dict())
        model.eval()
        model.to(device)
    else:
        model=rmodel
        model.eval()

    attack_lst=list(default_attacks.values())
    success_img = 0
    while 1:
        filename_lst,attack_type=taskque.get()
        for filename in filename_lst:
            image_path = os.path.join(src_dir, filename)
            image=load_image(image_path,device)
            label=int(filename[0])
            attacki=attack_lst[attack_type]
            image=attacki(image,eps,alpha,iters,model,torch.tensor([label]).to(device))
            if image is not None:
                image=(image*255).to(torch.uint8)
                # 命名格式 <label>_<pic_id>_adv_<id>.png
                filename = f"{label}_{success_img}_adv_{id}.png"
                save_path = os.path.join(dst_dir, filename)
                save_image = transforms.ToPILImage()(image.squeeze(0))
                save_image.save(save_path)
                success_img += 1
        arr[id]+=1
    

default_attacks={'fgsm':fgsm_attack,'pgd':pgd_attack,'bim':bim_attack}

class Adv_generator:
    def __init__(self,target_dir:str, src_dir:str,eps:float=0.1,
                 alpha: float=0.005, iters: int=50, attacks:Optional[List[str]]=None) -> None:
        # self.model=ViolenceClassifier()
        self.target_dir=target_dir
        self.src_dir=src_dir
        self.eps=eps
        self.alpha=alpha
        self.iters=iters
        if attacks is None:
            self.attacks=default_attacks.values()
        else:
            self.attacks=[default_attacks[key] for key in attacks]
    
    def generate(self, model:ViolenceClassifier, num:List[int],remove:bool=True)-> List[int]:
        '''
        num: success number of samples generated from each attack method
        return: success generated samples
        '''
        if remove:
            self.remove_all()
        device=model.device
        file_lst=os.listdir(self.src_dir)
        pbar=tqdm(total=sum(num),desc="generate attacks")
        success_lst=[]
        for ind,attacki in enumerate(self.attacks):
            lst_ind=0
            success_img=0
            while success_img<num[ind] and lst_ind<len(file_lst):
                image=load_image(f"{self.src_dir}/{file_lst[lst_ind]}",device)
                label=int(file_lst[lst_ind][0])
                image=attacki(image,self.eps,self.alpha,self.iters,model,torch.tensor([label]).to(device))
                if image is not None:
                    image=(image*255).to(torch.uint8)
                    filename = file_lst[lst_ind].replace(".jpg", ".png")
                    # 命名格式 <label>_<pic_id>_adv.png
                    filename = f"{label}_{success_img}_adv.png"
                    save_path = os.path.join(self.target_dir, filename)
                    save_image = transforms.ToPILImage()(image.squeeze(0))
                    save_image.save(save_path)
                    success_img += 1
                    pbar.update(1)
                lst_ind+=1
            success_lst.append(success_img)
        return success_lst
    
    def remove_all(self):
        for r,d,f in os.walk(self.target_dir):
            for fi in f:
                if "_adv" in fi and ".png" in fi:
                    os.remove(f"{r}/{fi}")


if __name__ == '__main__':
    print(default_attacks.values())


    # # 解析命令行参数
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-model', type=str, help="path of model file *.ckpt")
    # parser.add_argument('-eps', type=float, help="maximum value of `epsilon`")
    # parser.add_argument('-num', type=int, default=-1, help="the number of samples")
    # parser.add_argument('-alpha', type=float, default=0.005, help="step size for each iteration")
    # parser.add_argument('-iters', type=int, default=50, help="number of iterations")
    # parser.add_argument('-j',type=int,default=2,help="num of multiprocess")
    # parser.add_argument('-test',type=str,default="../test",help="path of test")
    # args = parser.parse_args()
    # print(args)

    # test_directory = args.test
    # bim_directory = f'bim_eps={args.eps}'
    # epsilon = args.eps  # 最大扰动
    # alpha = args.alpha  # 每一步的扰动
    # iters = args.iters  # 迭代次数