import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from model import ViolenceClassifier
from dataset import CustomDataModule

model_weight_path="train_logs"


def adv_train(echo_step:int, num:int):
    '''
    adv_train: 每过`echo_step`加入对抗样本进入train数据中
    num: 每次加入train数据的样本数量
    '''
    return 0



if __name__ == '__main__':
    acc_device='gpu' if torch.cuda.is_available() else 'cpu'
    print(f"using {acc_device=}")
    acc_id = [0] if torch.cuda.is_available() else None
    lr = 3e-6
    batch_size = 128
    log_name = "resnet18_pretrain_test"
    print("{} {}: {}, batch size: {}, lr: {}".format(log_name,acc_device, acc_id, batch_size, lr))

    data_module = CustomDataModule(batch_size=batch_size,num_workers=4)
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

    # 实例化训练器
    trainer = Trainer(
        logger= logger,
        max_epochs=2000,
        accelerator=acc_device,
        devices=acc_id,
        callbacks=[checkpoint_callback,checkpoint_callback1],
    )

    # 实例化模型
    model = ViolenceClassifier(learning_rate=lr,l2_param=2e-3)
    # 开始训练
    trainer.fit(model, data_module)
