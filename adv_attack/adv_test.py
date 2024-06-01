import os
import torch
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from model import ViolenceClassifier
from dataset import CustomDataModule

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model',type=str,help="path of model file *.pth")
    parser.add_argument('-eps',type=float,help="the value of `ep`")
    parser.add_argument('--attacks',type=str,nargs='+',help="attacks to be tested, ex. `fgsm`")
    args = parser.parse_args()
    print(args)

    # 设备选择：自动检测是否使用 GPU
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    device_count = [0] if device == 'gpu' else None  # 对于 GPU 使用一个设备，CPU 不需要设定
    batch_size = 256
    log_name = "resnet18_pretrain"

    # 设置数据模块
    data_module = CustomDataModule(batch_size=batch_size)

    # 加载模型并创建训练器
    model = ViolenceClassifier()
    model.model.load_state_dict(torch.load(args.model))
    trainer = Trainer(accelerator=device, devices=device_count)

    attacks = args.attacks


    # 噪声测试集：guass，salt
    # 对抗测试集：FGSM, BIM, PGD, C&W ,FGSM+PGD
    
    for attack in attacks:
        attack=attack+f"_eps={args.eps}"
        if not os.path.exists(attack):
            print(f"skipped {attack.upper()} data")
            continue
        print(f"Testing on {attack.upper()} perturbed data")
        data_module.set_test_path(os.path.join(os.getcwd(), attack))
        trainer.test(model, data_module)
