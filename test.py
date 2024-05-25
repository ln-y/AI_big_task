import os
import torch
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from model import ViolenceClassifier
from dataset import CustomDataModule
from train import model_weight_path

if __name__ == '__main__':
    # 设备选择：自动检测是否使用 GPU
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    device_count = [1] if device == 'gpu' else None  # 对于 GPU 使用一个设备，CPU 不需要设定
    batch_size = 128
    log_name = "resnet18_pretrain"

    # 设置数据模块
    data_module = CustomDataModule(batch_size=batch_size)

    # 模型路径设置
    ckpt_path = os.getcwd() + '/model/resnet18_pretrain_test-epoch=10-val_loss=0.06.ckpt'
    logger = TensorBoardLogger("test_logs", name=log_name)

    # 加载模型并创建训练器
    model = ViolenceClassifier.load_from_checkpoint(ckpt_path)
    trainer = Trainer(accelerator=device, devices=device_count)

    # # 测试原始测试数据
    # print("Testing on original test data")
    # trainer.test(model, data_module)
    #
    # # 测试高斯噪声
    # print("Testing on gauss noise data")
    # data_module.set_test_path(os.path.join(os.getcwd(), 'gauss'))
    # trainer.test(model, data_module)
    #
    # # 测试椒盐噪声
    # print("Testing on Salt and Pepper Noise data")
    # data_module.set_test_path(os.path.join(os.getcwd(), 'salt'))
    # trainer.test(model, data_module)
    #
    # # 对抗测试集：FGSM, BIM, PGD, C&W
    # attacks = ['fgsm', 'bim', 'pgd', 'c_w']
    # for attack in attacks:
    #     print(f"Testing on {attack.upper()} perturbed data")
    #     data_module.set_test_path(os.path.join(os.getcwd(), attack))
    #     trainer.test(model, data_module)

    # 测试AIGC训练集
    print("Testing on AIGC pictures")
    images_dir = os.path.join(os.getcwd(), 'images')

    for subfolder in os.listdir(images_dir):

        subfolder_path = os.path.join(images_dir, subfolder)
        if os.path.isdir(subfolder_path):
            print(f"Testing on {subfolder} images")
            data_module.set_test_path(subfolder_path)
            trainer.test(model, data_module)


