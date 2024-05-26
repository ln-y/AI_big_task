import os
import numpy as np
from PIL import Image
from torchvision import transforms

transformer = transforms.Compose([
                transforms.ToTensor(),  # 将图像转换为Tensor
            ])

path="images/model4"
image_lst=os.listdir(path)
for fi in image_lst:
    x=Image.open(f"{path}/{fi}")
    y=transformer(x)
    assert y.shape[0]==3,f"{y.shape=}\n{np.array(x).shape=}\n{x.mode}"