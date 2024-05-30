import torch
from torch import nn
import numpy as np
from torchvision import models
from torchmetrics import Accuracy
from pytorch_lightning import LightningModule
from typing import Optional
# from fgsm import fgsm_attack

# attack_dic={'fgsm':fgsm_attack}


class ViolenceClassifier(LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-3, l2_param= 1e-5, adv_train_step:int=1):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        # self.model = models.resnet18(pretrained=False, num_classes=2)

        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失
        self.accuracy = Accuracy(task="multiclass", num_classes=2)
        self.l2_param=l2_param
        self.echo_id=0
        self.adv_train_step=adv_train_step
        self.ids=0

    def forward(self, x):
        # print("used")
        # breakpoint()
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_param)  # 定义优化器
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        return loss
    
    def on_train_epoch_end(self) -> None:
        self.echo_id+=1
        # if self.echo_id%self.adv_train_step:

        # print(f"hello:{self.echo_id}")
        return super().on_epoch_end()


# def generate_adv_sample(num:int,algorithms:Optional[list],):
#     '''
#     num: generate success samples of each algorithm 
#     '''
#     if algorithms is None:
#         algorithms=['fgsm']
#     for algorithm in algorithms:
#         attack_func=attack_dic[algorithm]
