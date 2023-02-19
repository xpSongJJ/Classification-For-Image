import torch.utils.data
from config import Config
from model import GoogLeNet as create_model
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
import torch.optim as optim
import os
from tqdm import tqdm
import sys


def main():
    config = Config()
    print("using {} device".format(config.device))

    writer = SummaryWriter()
    data_transforms = {
        "train": transforms.Compose([transforms.Resize(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

    train_dataset = datasets.ImageFolder(root=os.path.join(config.data_root, "train", config.snr[9]),
                                         transform=data_transforms["train"])
    val_dataset = datasets.ImageFolder(root=os.path.join(config.data_root, "val", config.snr[9]),
                                       transform=data_transforms["val"])
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

    net = create_model(num_classes=config.classes, aux_logits=True, init_weights=True)
    print("Total number of parameters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))
    if os.path.exists(config.weights_file):
        net.load_state_dict(torch.load(config.weights_file))

    net.to(config.device)
    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)
    best_acc = 0.0
    for epoch in range(config.epochs):
        # train
        net.train()
        train_loss = 0.0
        train_acc_num = 0
        train_acc = 0
        sample_num = 0
        n_batch = 0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits, aux_logits2, aux_logits1 = net(images.to(config.device))
            loss0 = config.loss_function(logits, labels.to(config.device))
            loss1 = config.loss_function(aux_logits1, labels.to(config.device))
            loss2 = config.loss_function(aux_logits2, labels.to(config.device))
            loss = loss0 + loss1 * 0.3 + loss2 * 0.3

            predict_labels = torch.max(logits, dim=1)[1]
            train_acc_num += torch.eq(predict_labels, labels.to(config.device)).sum()
            sample_num += images.shape[0]
            loss.backward()
            optimizer.step()
            n_batch = step + 1
            # print statistics
            train_loss += loss.item()
            train_acc = train_acc_num / sample_num

        print("epoch: {}/{}, train_acc: {:.3f}, train_loss: {:.3f}".format(epoch,
                                                                           config.epochs,
                                                                           train_acc,
                                                                           train_loss/n_batch))

        # valuate
        net.eval()
        val_acc_num = 0.0
        val_loss = 0.0
        n_batch = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(config.device))  # eval model only have last output layer
                predict_y = torch.max(outputs, dim=1)[1]
                val_acc_num += torch.eq(predict_y, val_labels.to(config.device)).sum().item()
                val_loss += config.loss_function(outputs, val_labels.to(config.device))
                n_batch += 1
        val_acc = val_acc_num / config.num_val
        val_loss = val_loss / n_batch
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
