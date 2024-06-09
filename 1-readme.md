## 环境配置
由于版本与大作业指导手册同步，采用了较老的版本，环境配置较为特殊，请遵循以下操作：

**linux**
```bash
#进入到对应的python环境
source runme.sh
```

**windows**
```bash
pip install -r requirements.txt
pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
pip install numpy==1.19.0
```

## classfy.py 接口调用实例说明
### 1. 导入必要的库以及模型路径设置

```python
import torch
import sys
from torchvision import transforms
from PIL import Image
import os

model_path = os.path.join(os.getcwd(), '1-其他支持文件和目录')  #设置模型路径
if model_path not in sys.path:
    sys.path.append(model_path)

from model import ViolenceClassifier
```

### 2. classify 接口类实现

```python
class ViolenceClass:
    def __init__(self, model_path: str, device: str = 'cuda:0'):
        # 选择设备：如果有 GPU 则使用 GPU，否则使用 CPU
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = ViolenceClassifier()  # 实例化模型
        # 加载模型权重
        self.model.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)  # 将模型移动到相应的设备上
        self.model.eval()  # 设置模型为评估模式

    def classify(self, imgs: torch.Tensor) -> list:
        imgs = imgs.to(self.device)  # 将输入张量移动到相应的设备上
        with torch.no_grad():  # 禁用梯度计算
            output = self.model(imgs)
            _, pred = torch.max(output, 1)
        return pred.tolist()  # 返回预测结果列表

```

#### 参数

- `model_path` (str): 预训练模型权重文件的路径。
- `device` (str): 运行模型的设备。默认值为`'cuda:0'`（GPU）。如果没有可用的GPU，它将退回到CPU。
- `imgs` (torch.Tensor): 一批图像，形状为(N, C, H, W)的四维张量，其中N为图像数量，C为通道数，H为高度，W为宽度。

#### 返回

- `list`: 预测结果列表，每个预测结果对应于批次中的一个图像。

### 3. 调用实例

在classify.py的main函数中，给出了如何从测试集中加载图像，并且转化为tensor张量，然后调用classfy函数进行分类。

#### 定义辅助函数加载图像

定义一个辅助函数`load_images_from_folder`，用于从指定文件夹中加载图像并进行预处理：

```python
def load_images_from_folder(folder: str, transform) -> torch.Tensor:
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            image = Image.open(img_path).convert("RGB")
            image = transform(image)
            images.append(image)
    if len(images) == 0:
        raise ValueError(f"No images found in folder: {folder}")
    return torch.stack(images)
```

- `folder` (str)：图像文件夹的路径。
- `transform`：图像预处理的转换函数。
- 该函数遍历文件夹中的所有图像文件，加载并转换为RGB格式，然后应用预定义的图像转换，将图像添加到列表中，最后将所有图像堆叠为一个四维张量返回。

#### 定义图像转换

定义一个图像转换序列，将图像调整为224x224像素并转换为张量：

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
```

- `transforms.Resize((224, 224))`：调整图像大小为224x224像素。
- `transforms.ToTensor()`：将图像转换为张量格式。

#### 加载图像

使用定义的辅助函数从指定文件夹加载图像：

```
test_folder = '1-其他支持文件和目录/test'
batch_images = load_images_from_folder(test_folder, transform)
```

- `test_folder` (str)：测试图像文件夹的路径。
- `batch_images`：加载并转换后的图像批次。

#### 初始化分类器

创建一个`ViolenceClass`实例，加载预训练的模型权重：

```python
classifier = ViolenceClass(model_path="1-其他支持文件和目录/model/ed2.pth")
```

- `model_path` (str)：预训练模型权重文件的路径。

#### 对图像进行分类

使用分类器对加载的图像批次进行分类，并打印预测结果：

```python
predictions = classifier.classify(batch_images)
print(predictions)
```

- `classifier.classify(batch_images)`：对图像批次进行分类，返回预测结果列表。
- `print(predictions)`：打印预测结果列表。
