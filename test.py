import os, shutil
import torch
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from model import ViolenceClassifier
from dataset import CustomDataModule
import argparse
from denoise import process_images

def eval_dir(target_dir:str,use_denoise):
    print(f"testing on {target_dir.upper()} data")
    if use_denoise:
        os.makedirs(f"{target_dir}/denoise")
        process_images(target_dir,f"{target_dir}/denoise")
        data_module.set_test_path(f"{target_dir}/denoise")
        trainer.test(model,data_module)
        shutil.rmtree(f"{target_dir}/denoise")
    else:
        if os.path.exists(f"{target_dir}/denoise"):
            shutil.rmtree(f"{target_dir}/denoise")
        data_module.set_test_path(target_dir)
        trainer.test(model,data_module)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-model',type=str,help="model path")
    parser.add_argument('--denoise',type=int,default=False)
    args=parser.parse_args()
    print(args)

    # 设备选择：自动检测是否使用 GPU
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    device_count = [0] if device == 'gpu' else None  # 对于 GPU 使用一个设备，CPU 不需要设定
    batch_size = 128
    log_name = "resnet18_pretrain"

    # 设置数据模块
    data_module = CustomDataModule(batch_size=batch_size)

    # 模型路径设置
    ckpt_path = os.getcwd() +'/'+ args.model
    logger = TensorBoardLogger("test_logs", name=log_name)

    # 加载模型并创建训练器
    model = ViolenceClassifier()
    model.model.load_state_dict(torch.load(args.model))
    trainer = Trainer(accelerator=device, devices=device_count)

    # 测试原始测试数据
    eval_dir("test",args.denoise)
    eval_dir("aigc_test",args.denoise)
    eval_dir("contrast_adv",args.denoise)
    eval_dir("noise_test",args.denoise)
    # eval_dir("all_test",args.denoise)

    # 对抗测试集：FGSM, BIM, PGD, C&W
    # attacks = ['fgsm', 'bim', 'pgd', 'c_w']
    # for attack in attacks:
    #     if not os.path.exists(attack):
    #         print(f"skipped {attack.upper()} data")
    #         continue
    #     print(f"Testing on {attack.upper()} perturbed data")
    #     data_module.set_test_path(os.path.join(os.getcwd(), attack))
    #     trainer.test(model, data_module)

    # aigc_img_path="images"
    # for r,d,f in os.walk(aigc_img_path):
    #     for dir in d:
    #         dir=f"{aigc_img_path}/{dir}"
    #         print(f"Testing on {dir} aigc data")
    #         data_module.set_test_path(os.path.join(os.getcwd(), dir))
    #         trainer.test(model, data_module)
