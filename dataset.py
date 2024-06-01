from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from tqdm import tqdm
import functools

def load_image(image_path,device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    image = torch.from_numpy(image).permute(2, 0, 1).to(device).contiguous()  # 调整维度顺序
    image = (image/255).to(torch.float)
    return image

class CustomDataset(Dataset):
    def __init__(self, split, data_root=None,device=None):
        if device is None:
            self.device='cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device=device
        assert split in ["train", "val", "test"]
        self.split=split
        if split == "train":
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),  # 随机翻转
                # transforms.ToTensor(),  # 将图像转换为Tensor
            ])
        else:
            self.transforms = transforms.Compose([
                # transforms.ToTensor(),  # 将图像转换为Tensor
            ])
        if data_root is not None:
            split_path=os.path.join(os.getcwd(),data_root)
        else:
            split_path = os.path.join(os.getcwd(), split)
        self.data = [os.path.join(split_path, i) for i in os.listdir(split_path)]
        
        # self.cache_lst=[self.get_cache(i) for i in tqdm(range(len(self.data)),desc="cache_pictures")]
        self.time_record=0
        
    
    def get_cache(self, index):
        img_path = self.data[index]
        x = Image.open(img_path).convert("RGB")
        y = int(os.path.basename(img_path)[0]) # 获取标签值，0代表非暴力，1代表暴力
        x = self.transforms(x)
        if self.split=="train":
            x=x.to(self.device)
        return x, y
    
    def __len__(self):
        return len(self.data)

    # @functools.lru_cache(40000)
    def __getitem__(self, index):
        img_path = self.data[index]
        x = load_image(img_path,self.device)
        y = int(os.path.basename(img_path)[0]) # 获取标签值，0代表非暴力，1代表暴力
        x = self.transforms(x)
        return x,y
        

class CustomDataModule(LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4,test_path='test',root_path="."):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_path = f"{root_path}/{test_path}"
        self.root_path = root_path


    def setup(self, stage=None):
        # 分割数据集、应用变换等
        # 创建 training, validation数据集
        if stage=="test":
            self.test_dataset = CustomDataset(f"{self.root_path}/test", data_root=self.test_path)
        else:
            self.train_dataset = CustomDataset("train",data_root=f"{self.root_path}/train")
            self.val_dataset = CustomDataset("val",data_root=f"{self.root_path}/val")
            self.test_dataset = CustomDataset("test", data_root=self.test_path)

    def train_dataloader(self):
        # print(f"train called")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        # print(f"val called")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
    def test_dataloader(self):
        # print(f"test called")
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def set_test_path(self, path):
        self.test_path = path
        self.setup(stage='test')