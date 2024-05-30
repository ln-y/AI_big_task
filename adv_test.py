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

    from torchvision import transforms
    from PIL import Image
    from torchmetrics import Accuracy
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    # x=torch.load('0.pt')
    # y=torch.load('1.pt')
    # print(f"{model(x)=}\n{model(y)=}")
    # print(f"{model.test_step((x,torch.tensor([0,0,0,0])),0)}\n{model.test_step((y,torch.tensor([0,0,0,0])),0)}")

    paths=os.path.join(os.getcwd(), attacks[0]+f"_eps={args.eps}")
    im_lst=[]
    for ind,pic in enumerate(os.listdir(paths)):
        preprocess = transforms.Compose([
        transforms.ToTensor(),
        ])
        image = Image.open(f"{paths}/{pic}").convert("RGB")
        image = preprocess(image)
        im_lst.append(image)
    image=torch.stack(im_lst)
    print(image.shape)    
    predictions=trainer.predict(model,dataloaders=DataLoader(image))
    print(predictions)
    # logits=model.test_step((image,torch.tensor([0,0,0,0])),0)
    # # breakpoint()
    # print(logits)
    # print(image)
    # if ind==1:
    #     torch.save(image,"2.pt")
    # accf=Accuracy(task="multiclass", num_classes=2)
    # loss_fn = nn.CrossEntropyLoss()
    # print(accf(logits,y),loss_fn(logits,y))