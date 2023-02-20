import math
import os
import random
import re
import shutil

import numpy as np
import torch
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from tqdm import tqdm
import sys


def try_a_train(model, optimizer, loss_function, data_loader, device, lr_scheduler=None):
    model.train()
    train_acc_num = torch.zeros(1).to(device)
    train_loss = torch.zeros(1).to(device)
    sample_num = 0

    optimizer.zero_grad()
    data_loader = tqdm(data_loader, file=sys.stdout)
    n_batch = 0
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        outputs = model(images.to(device))
        predict_labels = torch.max(outputs, dim=1)[1]
        train_acc_num += torch.eq(predict_labels, labels.to(device)).sum()
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        train_loss += loss.detach()

        optimizer.step()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()
        n_batch = step

    return train_acc_num.item() / sample_num, train_loss.item() / n_batch


@torch.no_grad()
def try_an_evaluate(model, loss_function, data_loader, device):
    model.eval()

    val_acc_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    val_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    n_batch = 0
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        outputs = model(images.to(device))
        predict_labels = torch.max(outputs, dim=1)[1]
        val_acc_num += torch.eq(predict_labels, labels.to(device)).sum()

        loss = loss_function(outputs, labels.to(device))
        val_loss += loss
        n_batch = step
    return val_acc_num.item() / sample_num, val_loss.item() / n_batch


def split_data(root: str, target_root: str, snr: str, ratio: float):
    """

    :param root: 源数据根目录
    :param target_root: 目标文件根目录
    :param snr: 信噪比文件名称
    :param ratio: 测试集占比
    :return: None
    """

    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    os.makedirs(os.path.join(target_root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(target_root, 'val'), exist_ok=True)
    train_root = os.path.join(target_root, 'train', snr)
    val_root = os.path.join(target_root, 'val', snr)

    snr_root = os.path.join(root, snr)

    class_names = os.listdir(snr_root)
    pattern = re.compile(r'.*\..*')
    class_names = [name for name in class_names if not pattern.match(name)]
    for class_name in class_names:
        os.makedirs(os.path.join(train_root, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_root, class_name), exist_ok=True)
        cla_root = os.path.join(snr_root, class_name)
        img_names = os.listdir(cla_root)
        img_num = len(img_names)
        split_img_names = random.sample(img_names, int(img_num*ratio))
        for img_name in img_names:
            if img_name in split_img_names:
                shutil.copy(os.path.join(cla_root, img_name),
                            os.path.join(val_root, class_name, img_name))
            else:
                shutil.copy(os.path.join(cla_root, img_name),
                            os.path.join(train_root, class_name, img_name))


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()
