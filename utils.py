import os
import random
import re
import shutil

import torch
from tqdm import tqdm
import sys


def try_a_train(model, optimizer, loss_function, data_loader, device):
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
