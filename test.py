import os
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from model import ViolenceClassifier
from dataset import CustomDataModule

from train import model_weight_path

acc_device='gpu' if torch.cuda.is_available() else 'cpu'
print(f"using {acc_device=}")
acc_id = [1] if torch.cuda.is_available() else None
batch_size = 128
log_name = "resnet18_pretrain"

data_module = CustomDataModule(batch_size=batch_size)
ckpt_root = f"{os.getcwd()}/{model_weight_path}"
ckpt_path = ckpt_root + "/resnet18_pretrain_test/version_0/checkpoints/resnet18_pretrain_test-epoch=32-val_loss=0.03.ckpt"
logger = TensorBoardLogger("test_logs", name=log_name)

model = ViolenceClassifier.load_from_checkpoint(ckpt_path)
trainer = Trainer(accelerator=acc_device, devices=acc_id)
trainer.test(model, data_module)