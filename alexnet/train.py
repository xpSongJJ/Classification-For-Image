import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = "../../data/image"
    snr = '10dB_tf'
    train_dataset = datasets.ImageFolder(root=os.path.join(data_root, "train", snr),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    cla_dict = dict((val, key) for key, val in train_dataset.class_to_idx.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size, shuffle=True,
                                                    num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(data_root, "val", snr),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    val_data_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = AlexNet(num_classes=12, init_weights=True)
    if os.path.exists('AlexNet.pth'):
        net.load_state_dict(torch.load("AlexNet.pth"))

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    tb_writer = SummaryWriter()
    epochs = 100
    save_path = './AlexNet.pth'
    best_acc = 0.0
    for epoch in range(epochs):
        # train
        net.train()
        train_acc = 0.0
        training_loss = 0.0
        train_loss = 0.0
        train_bar = tqdm(train_data_loader, file=sys.stdout)
        acc_num = torch.zeros(1).to(device)
        sample_num = torch.zeros(1).to(device)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            prediction = torch.max(outputs, dim=1)[1]
            acc_num += torch.eq(prediction, labels.to(device)).sum()
            sample_num += images.shape[0]
            train_acc = acc_num / sample_num
            training_loss += loss.item()
            train_loss = training_loss / sample_num

            train_bar.desc = "train epoch[{}/{}] acc:{:.3f} loss:{:.3f}".format(epoch + 1,
                                                                                epochs,
                                                                                train_acc.item(),
                                                                                train_loss.item())

        # validate
        net.eval()
        val_acc = torch.zeros(1).to(device)
        val_total_loss = torch.zeros(1).to(device)
        val_loss = torch.zeros(1).to(device)
        val_acc_num = torch.zeros(1).to(device)
        sample_num = 0
        with torch.no_grad():
            val_bar = tqdm(val_data_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                val_prediction = torch.max(outputs, dim=1)[1]
                val_acc_num += torch.eq(val_prediction, val_labels.to(device)).sum().item()
                sample_num += val_images.shape[0]
                val_acc = val_acc_num / sample_num
                val_total_loss += loss_function(outputs, val_labels.to(device))
                val_loss = val_total_loss / sample_num

                val_bar.desc = "valid epoch[{}/{}] acc:{:.3f} loss:{:.3f}".format(epoch + 1,
                                                                                  epochs,
                                                                                  val_acc.item(),
                                                                                  val_loss.item())

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)

        tb_writer.add_scalar("train_acc", train_acc, epoch)
        tb_writer.add_scalar("train_loss", train_loss, epoch)
        tb_writer.add_scalar("val_acc", val_acc, epoch)
        tb_writer.add_scalar("val_loss", val_loss, epoch)

    print('Finished Training')


if __name__ == '__main__':
    main()
