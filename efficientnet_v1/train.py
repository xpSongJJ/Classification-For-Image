import math

import torch.utils.data
from config import Config
from model import efficientnet_b0 as create_model
from utils import try_a_train, try_an_evaluate
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os
from torch.utils.data import ConcatDataset


def main():
    config = Config()
    print("using {} device".format(config.device))
    config.weights_file = 'efficientnet_b0.pth'

    writer = SummaryWriter()
    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}

    data_transforms = {
        "train": transforms.Compose([transforms.Resize(img_size["B0"]),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize(img_size["B0"]),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

    # ConcatDataSet from different snr
    train_dataset = datasets.ImageFolder(root=os.path.join(config.data_root, "train", config.snr[10]),
                                         transform=data_transforms["train"])
    val_dataset = datasets.ImageFolder(root=os.path.join(config.data_root, "val", config.snr[10]),
                                       transform=data_transforms["val"])
    for i in range(10):
        train_ds = datasets.ImageFolder(root=os.path.join(config.data_root, "train", config.snr[i]),
                                        transform=data_transforms["train"])
        val_ds = datasets.ImageFolder(root=os.path.join(config.data_root, "val", config.snr[i]),
                                      transform=data_transforms["val"])
        train_dataset = ConcatDataset([train_dataset, train_ds])
        val_dataset = ConcatDataset([val_dataset, val_ds])
    config.num_train = len(train_dataset)
    config.num_val = len(val_dataset)
    print("train data number: {}, valuate data number: {}".format(config.num_train, config.num_val))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               config.batch_size,
                                               shuffle=True,
                                               num_workers=config.n_works)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=True,
                                             num_workers=config.n_works)
    print("using {} subprocess for data loading".format(config.n_works))

    net = create_model(num_classes=config.classes)
    print("Total number of parameters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))
    if os.path.exists(config.weights_file):
        net.load_state_dict(torch.load(config.weights_file))

    net.to(config.device)
    optimizer = optim.SGD(net.parameters(), lr=config.learning_rate, weight_decay=1E-4)
    lf = lambda x: ((1 + math.cos(x * math.pi / config.epochs)) / 2) * (1 - 0.01) + 0.01  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_acc = 0.0
    for epoch in range(config.epochs):
        # train
        train_acc, train_loss = try_a_train(model=net,
                                            optimizer=optimizer,
                                            loss_function=config.loss_function,
                                            data_loader=train_loader,
                                            device=config.device)
        scheduler.step()
        print("epoch: {}/{}, train_acc: {:.3f}, train_loss: {:.3f}".format(epoch, config.epochs, train_acc, train_loss))

        # valuate
        val_acc, val_loss = try_an_evaluate(model=net,
                                            loss_function=config.loss_function,
                                            data_loader=val_loader,
                                            device=config.device)
        print("epoch: {}/{}, val_acc: {:.3f}, val_loss: {:.3f}".format(epoch, config.epochs, val_acc, val_loss))

        writer.add_scalar("train_acc", train_acc, epoch)
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("val_acc", val_acc, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)

        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), config.weights_file)

    print("training end")


if __name__ == '__main__':
    main()
