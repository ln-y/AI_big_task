import json

import torch
from model import ViolenceClassifier
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm  # 引入 tqdm 库

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
        preds = []
        with torch.no_grad():  # 禁用梯度计算
            for i in tqdm(range(len(imgs)), desc="Classifying"):  # 添加进度条
                output = self.model(imgs[i].unsqueeze(0))  # 对每张图像进行推理
                _, pred = torch.max(output, 1)  # 获取预测类别
                preds.append(pred.item())
        return preds  # 返回预测结果列表

## 以下为试验代码，实际不需要
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

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 加载 'test' 文件夹中的图片
    test_folder = 'test'
    batch_images = load_images_from_folder(test_folder, transform)

    # 使用 ViolenceClass 进行分类
    classifier = ViolenceClass(model_path="model/ed2_1_acc.pth")
    predictions = classifier.classify(batch_images)
    with open(f"classify.py.json", "w") as f:
        json.dump(predictions, f)
    print(predictions)
