import os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from model import ViolenceClassifier
from dataset import CustomDataModule

from train import model_weight_path

gpu_id = [0]
batch_size = 128
log_name = "resnet18_pretrain"

data_module = CustomDataModule(batch_size=batch_size)
ckpt_root = f"{os.getcwd()}/{model_weight_path}"
ckpt_path = ckpt_root + "/resnet18_pretrain_test/version_0/checkpoints/resnet18_pretrain_test-epoch=32-val_loss=0.03.ckpt"
logger = TensorBoardLogger("test_logs", name=log_name)

model = ViolenceClassifier.load_from_checkpoint(ckpt_path)
trainer = Trainer(accelerator='gpu', devices=gpu_id)
trainer.test(model, data_module)