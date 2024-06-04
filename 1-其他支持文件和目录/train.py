import torch
import os, shutil, subprocess
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm
from model import ViolenceClassifier
from dataset import CustomDataModule
from noise_and_attack.adv_attack import Adv_generator
from typing import Optional,Tuple

model_weight_path="train_logs"

class AdvTrainModel(ViolenceClassifier):
    def __init__(self, num_classes=2, learning_rate=0.001, l2_param=0.00001, adv_train_step: int = 1,
                 target_dir:str="train", src_dir:str="train", adv_num:Tuple= (100,100,100)):
        self.adv_train_step=adv_train_step
        self.target_dir=target_dir
        self.src_dir=src_dir
        self.adv_generator=Adv_generator(target_dir,src_dir)
        self.adv_generator.remove_all()
        self.echo_id=0
        self.adv_num=list(adv_num)
        self.data_module:Optional[CustomDataModule]=None
        super().__init__(num_classes, learning_rate, l2_param)

    def on_train_epoch_end(self) -> None:
        self.echo_id+=1
        
        if self.echo_id%self.adv_train_step==0:
            self.adv_generator.generate(self,self.adv_num)
            assert self.data_module is not None
            self.data_module.update_train()
        return super().on_epoch_end()



if __name__ == '__main__':
    adv_step= 3
    adv_num= (300,1000,1000)
    total_cycle = 600
    acc_device='gpu' if torch.cuda.is_available() else 'cpu'
    print(f"using {acc_device=}")
    acc_id = [0] if torch.cuda.is_available() else None
    lr = 1e-5
    batch_size = 128
    log_name = "resnet18_pretrain_test"
    print("{} {}: {}, batch size: {}, lr: {}".format(log_name,acc_device, acc_id, batch_size, lr))

    data_module = CustomDataModule(batch_size=batch_size,num_workers=0)
    # 设置模型检查点，用于保存最佳模型
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename=log_name + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    checkpoint_callback1=ModelCheckpoint(
        monitor='val_acc',
        filename=log_name + '-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1,
        mode='max',
    )
    logger = TensorBoardLogger(model_weight_path, name=log_name)


    trainer_dic={"logger": logger,
        "max_epochs":adv_step,
        "accelerator":acc_device,
        "devices":acc_id,
        "callbacks":[checkpoint_callback,checkpoint_callback1],}
    # 实例化训练器
    

    # 实例化模型
    model = AdvTrainModel(learning_rate=lr,l2_param=1e-4,adv_train_step=adv_step,adv_num=adv_num)
    model.data_module=data_module
    # 开始训练
    for _ in tqdm(range(total_cycle),desc="total cycle"):
        trainer = Trainer(
        **trainer_dic
        )
        trainer.fit(model, data_module)
    
