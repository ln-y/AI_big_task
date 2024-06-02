#这个完全在cpu上面跑，可以运行了

import os
import torch
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from model import ViolenceClassifier
from dataset import CustomDataModule
from train import model_weight_path

import numpy as np
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

import argparse

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("-model",type=str)
    args=parser.parse_args()
    print(args)    

    # 设备选择：自动检测是否使用 GPU，如果没有GPU，则使用CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'   # 直接指定使用CPU
    print(f"Using device: {device}")

    batch_size = 128
    log_name = "resnet18_pretrain"

    # 设置数据模块
    data_module = CustomDataModule(batch_size=batch_size)

    # # 模型路径设置
    # ckpt_path = os.path.join(os.getcwd(), 'model', 'resnet18_pretrain_test-epoch=10-val_loss=0.06.ckpt')
    # # 加载模型并将其移动到相应的设备
    # model = ViolenceClassifier.load_from_checkpoint(ckpt_path).to(device)
    # 之后的模型将使用.pth文件的形式发布，使用以下代码加载
    model=ViolenceClassifier()
    model.model.load_state_dict(torch.load(args.model)) # 假设ed1.pth即为对应的pth文件


    logger = TensorBoardLogger("test_logs", name=log_name)

    # 创建训练器
    trainer = Trainer(accelerator='gpu' if device=="cuda" else device, devices=[0] if device=="cuda" else 1, logger=logger)

    # 测试数据集的名称 #['contrast_adv']#
    datasets =  ['contrast_adv']#['fgsm_eps=0.03','cw_eps=0.03',"bim_eps=0.03","pgd_eps=0.03"]
    # 初始化变量来存储结果
    all_roc_aucs = []
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    all_confusion_matrices = []

    for dataset_name in datasets:
        # 根据数据集名称设置数据模块的路径
        data_module.set_test_path(os.path.join(os.getcwd(), dataset_name))

        # 测试模型
        print(f"Testing on {dataset_name} data")
        trainer.test(model, data_module)

        # 收集数据
        test_loader = data_module.test_dataloader()
        model.eval()  # 将模型设置为评估模式
        model.to(device)
        true_labels_list = []
        predicted_probs_list = []
        with torch.no_grad():
            for x, y in test_loader:
                # 确保输入数据和模型都在同一个设备上
                x = x.to(device)
                outputs = model(x)
                # 选择正类概率
                probabilities = outputs[:, 1].cpu().numpy()
                y_labels = y.view(-1).cpu().numpy()  # 确保 y 是一维的
                true_labels_list.extend(y_labels)
                predicted_probs_list.extend(probabilities)

        true_labels = np.array(true_labels_list)
        predicted_probs = np.array(predicted_probs_list).squeeze()

        # 计算 ROC 曲线的 FPR 和 TPR
        fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
        roc_auc = auc(fpr, tpr)
        all_roc_aucs.append(roc_auc)

        # 计算精确度、召回率和F1分数
        precision = precision_score(true_labels, (predicted_probs > 0.5).astype(int))
        recall = recall_score(true_labels, (predicted_probs > 0.5).astype(int))
        f1 = f1_score(true_labels, (predicted_probs > 0.5).astype(int))
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1_scores.append(f1)

        # 打印指标
        print(f"Metrics for {dataset_name}:")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        # 绘制ROC曲线
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{dataset_name} ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(f"test_{dataset_name}.png")
        plt.cla()

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(true_labels, (predicted_probs > 0.5).astype(int))
        all_confusion_matrices.append(conf_matrix)
        print("Confusion Matrix:\n", conf_matrix)

    print("\nAll ROC AUCs:", all_roc_aucs)
    print("All Precisions:", all_precisions)
    print("All Recalls:", all_recalls)
    print("All F1 Scores:", all_f1_scores)
    # Optionally, you can also print all confusion matrices
    # for cm in all_confusion_matrices:
    #     print(cm)