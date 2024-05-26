import os
import torch
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from model import ViolenceClassifier
from dataset import CustomDataModule

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model',type=str,help="path of model file *.ckpt")
    parser.add_argument('-eps',type=float,help="the value of `ep`")
    parser.add_argument('--attacks',type=str,nargs='+',help="attacks to be tested, ex. `fgsm`")
    args = parser.parse_args()
    print(args)

    # 设备选择：自动检测是否使用 GPU
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    device_count = [0] if device == 'gpu' else None  # 对于 GPU 使用一个设备，CPU 不需要设定
    batch_size = 128
    log_name = "resnet18_pretrain"

    # 设置数据模块
    data_module = CustomDataModule(batch_size=batch_size)

    # 模型路径设置
    ckpt_path = os.getcwd() + '/'+args.model
    # logger = TensorBoardLogger("test_logs", name=log_name)

    # 加载模型并创建训练器
    model = ViolenceClassifier.load_from_checkpoint(ckpt_path)
    trainer = Trainer(accelerator=device, devices=device_count)


    # 对抗测试集：FGSM, BIM, PGD, C&W
    attacks = args.attacks
    for attack in attacks:
        attack=attack+f"_eps={args.eps}"
        if not os.path.exists(attack):
            print(f"skipped {attack.upper()} data")
            continue
        print(f"Testing on {attack.upper()} perturbed data")
        data_module.set_test_path(os.path.join(os.getcwd(), attack))
        trainer.test(model, data_module)

