import os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from model import ViolenceClassifier
from dataset import CustomDataModule

from train import model_weight_path
if __name__ == '__main__':
    gpu_id = [0]
    batch_size = 128
    log_name = "resnet18_pretrain"

    data_module = CustomDataModule(batch_size=batch_size)
    #ckpt_root = f"{os.getcwd()}/{model_weight_path}"
    #ckpt_path = ckpt_root + "/resnet18_pretrain_test/version_0/checkpoints/resnet18_pretrain_test-epoch=32-val_loss=0.03.ckpt"
    ckpt_path= os.getcwd() + '/model/resnet18_pretrain_test-epoch=10-val_loss=0.06.ckpt'
    logger = TensorBoardLogger("test_logs", name=log_name)

    model = ViolenceClassifier.load_from_checkpoint(ckpt_path)
    trainer = Trainer(accelerator='cpu', devices=1)

    # 测试原始测试数据
    print("Testing on original test data")
    trainer.test(model, data_module)

    # 切换至对抗样本并测试
    #FGSM攻击
    print("Testing on FGSM perturbed data")
    data_module.set_test_path(os.path.join(os.getcwd(), 'fgsm'))
    trainer.test(model, data_module)

    #BIM攻击
    print("Testing on BIM perturbed data")
    data_module.set_test_path(os.path.join(os.getcwd(), 'bim'))
    trainer.test(model, data_module)

    #PGD攻击
    print("Testing on PGD perturbed data")
    data_module.set_test_path(os.path.join(os.getcwd(), 'pgd'))
    trainer.test(model, data_module)

    #C&W攻击
    print("Testing on C&W perturbed data")
    data_module.set_test_path(os.path.join(os.getcwd(), 'c_w'))
    trainer.test(model, data_module)



