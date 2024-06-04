
import os
import torch
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from model import ViolenceClassifier
from dataset import CustomDataModule
from train import model_weight_path
import json  # 导入json库
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 设备选择：自动检测是否使用 GPU，如果没有GPU，则使用CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'   # 直接指定使用CPU
    print(f"Using device: {device}")

    batch_size = 128
    log_name = "resnet18_pretrain"

    # 设置数据模块
    data_module1 = CustomDataModule(batch_size=batch_size)
    data_module2 = CustomDataModule(batch_size=batch_size)

    # # 模型路径设置
    # ckpt_path = os.path.join(os.getcwd(), 'model', 'resnet18_pretrain_test-epoch=10-val_loss=0.06.ckpt')
    # # 加载模型并将其移动到相应的设备
    # model = ViolenceClassifier.load_from_checkpoint(ckpt_path).to(device)
    # 之后的模型将使用.pth文件的形式发布，使用以下代码加载
    model1=ViolenceClassifier()
    model1.model.load_state_dict(torch.load("../model/ed1.pth")) # 假设ed1.pth即为对应的pth文件

    model2 = ViolenceClassifier()
    model2.model.load_state_dict(torch.load("../model/ed2.pth"))  # 假设ed1.pth即为对应的pth文件

    logger = TensorBoardLogger("../test_logs", name=log_name)

    # 创建训练器
    trainer1 = Trainer(accelerator='gpu' if device=="cuda" else device, devices=[0] if device=="cuda" else 1, logger=logger)
    trainer2 = Trainer(accelerator='gpu' if device == "cuda" else device, devices=[0] if device == "cuda" else 1,
                       logger=logger)

    # 测试数据集的名称
    datasets = ['test','aigc_test', 'noise_test', 'contrast_adv_b_0.03_all']
    #datasets = ['test']


    #datasets = ['test', 'fgsm']

    # 初始化变量来存储结果
    all_roc_aucs1 = []
    all_precisions1 = []
    all_recalls1 = []
    all_f1_scores1 = []
    all_confusion_matrices1 = []

    # 初始化变量来存储结果
    all_roc_aucs2 = []
    all_precisions2 = []
    all_recalls2 = []
    all_f1_scores2 = []
    all_confusion_matrices2 = []

    for dataset_name in datasets:
        # # 根据数据集名称设置数据模块的路径
        # data_module1.set_test_path(os.path.join(os.getcwd(), dataset_name))
        # data_module2.set_test_path(os.path.join(os.getcwd(), dataset_name))
        # 获取当前工作目录的父目录
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

        # 拼接路径，将路径设置为上一级目录中的dataset_name
        dataset_path = os.path.join(parent_dir, dataset_name)

        data_module1.set_test_path(dataset_path)
        data_module2.set_test_path(dataset_path)
        # 测试模型
        print('\n\n\n')
        print(f"Testing on {dataset_name} data")
        print("Testing model1...")
        trainer1.test(model1, data_module1)
        print("Testing model2...")
        trainer2.test(model2, data_module2)

        # 收集数据
        test_loader1 = data_module1.test_dataloader()
        test_loader2 = data_module2.test_dataloader()
        model1.eval()  # 将模型设置为评估模式
        model1.to(device)

        model2.eval()  # 将模型设置为评估模式
        model2.to(device)

######
        true_labels_list1 = []
        predicted_probs_list1 = []
        with torch.no_grad():
            for x, y in test_loader1:
                # 确保输入数据和模型都在同一个设备上
                x = x.to(device)
                outputs = model1(x)
                # 选择正类概率
                probabilities = outputs[:, 1].cpu().numpy()
                y_labels = y.view(-1).cpu().numpy()  # 确保 y 是一维的
                true_labels_list1.extend(y_labels)
                predicted_probs_list1.extend(probabilities)

        true_labels1 = np.array(true_labels_list1)
        predicted_probs1 = np.array(predicted_probs_list1).squeeze()

        # 计算 ROC 曲线的 FPR 和 TPR
        fpr1, tpr1, _ = roc_curve(true_labels1, predicted_probs1)
        roc_auc1 = auc(fpr1, tpr1)
        all_roc_aucs1.append(roc_auc1)

        # 计算精确度、召回率和F1分数
        precision1 = precision_score(true_labels1, (predicted_probs1 > 0.5).astype(int))
        recall1 = recall_score(true_labels1, (predicted_probs1 > 0.5).astype(int))
        f1_1 = f1_score(true_labels1, (predicted_probs1 > 0.5).astype(int))
        all_precisions1.append(precision1)
        all_recalls1.append(recall1)
        all_f1_scores1.append(f1_1)

        # 打印指标
        print(f"in model1: Metrics for {dataset_name}:")
        print(f"Precision: {precision1:.4f}, Recall: {recall1:.4f}, F1 Score: {f1_1:.4f}")
######

        true_labels_list2 = []
        predicted_probs_list2 = []
        #predicted_labels_list2 = []  # 用于存储预测标签
        with torch.no_grad():
            for x, y in test_loader2:
                # 确保输入数据和模型都在同一个设备上
                x = x.to(device)
                outputs = model2(x)
                # 选择正类概率
                probabilities = outputs[:, 1].cpu().numpy()
                y_labels = y.view(-1).cpu().numpy()  # 确保 y 是一维的
                true_labels_list2.extend(y_labels)
                predicted_probs_list2.extend(probabilities)

                # # 获取预测标签
                # _, preds = torch.max(outputs, 1)
                # predicted_labels_list2.extend(preds.cpu().tolist())

        true_labels2 = np.array(true_labels_list2)
        predicted_probs2 = np.array(predicted_probs_list2).squeeze()

        # # 将 predicted_labels_list2 保存到本地文件
        # with open(f"predicted_labels_{dataset_name}.json", "w") as f:
        #     json.dump(predicted_labels_list2, f)

        # 计算 ROC 曲线的 FPR 和 TPR
        fpr2, tpr2, _ = roc_curve(true_labels2, predicted_probs2)
        roc_auc2 = auc(fpr2, tpr2)
        all_roc_aucs2.append(roc_auc2)

        # 计算精确度、召回率和F1分数
        precision2 = precision_score(true_labels2, (predicted_probs2 > 0.5).astype(int))
        recall2 = recall_score(true_labels2, (predicted_probs2 > 0.5).astype(int))
        f1_2 = f1_score(true_labels2, (predicted_probs2 > 0.5).astype(int))
        all_precisions2.append(precision2)
        all_recalls2.append(recall2)
        all_f1_scores2.append(f1_2)

        # 打印指标
        print(f"in model2: Metrics for {dataset_name}:")
        print(f"Precision: {precision2:.4f}, Recall: {recall2:.4f}, F1 Score: {f1_2:.4f}")


    ####

        # 绘制ROC曲线
        plt.figure(figsize=(8, 6))
        plt.plot(fpr1, tpr1,  lw=2, label=f'model1:{dataset_name} ROC curve (AUC = {roc_auc1:.2f})')
        plt.plot(fpr2, tpr2,  lw=2, label=f'model2:{dataset_name} ROC curve (AUC = {roc_auc2:.2f})')
        plt.plot([0, 1], [0, 1],  lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

        # 计算混淆矩阵
        conf_matrix1 = confusion_matrix(true_labels1, (predicted_probs1 > 0.5).astype(int))
        all_confusion_matrices1.append(conf_matrix1)
        print("model1 Confusion Matrix:\n", conf_matrix1)

        conf_matrix2 = confusion_matrix(true_labels2, (predicted_probs2 > 0.5).astype(int))
        all_confusion_matrices2.append(conf_matrix2)
        print("model2 Confusion Matrix:\n", conf_matrix2)


    print("\nin model1:\n")
    print("All ROC AUCs:", all_roc_aucs1)
    print("All Precisions:", all_precisions1)
    print("All Recalls:", all_recalls1)
    print("All F1 Scores:", all_f1_scores1)

    print("\nin model2:\n")
    print("\nAll ROC AUCs:", all_roc_aucs2)
    print("All Precisions:", all_precisions2)
    print("All Recalls:", all_recalls2)
    print("All F1 Scores:", all_f1_scores2)

