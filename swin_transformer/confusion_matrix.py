import os
import json
import sys
import torch
from torch.utils.data import ConcatDataset
from torchvision import transforms, datasets
from tqdm import tqdm
from model import swin_tiny_patch4_window7_224 as create_model
from utils import ConfusionMatrix
from config import Config


def main():
    config = Config()
    # read class_indict
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=config.classes, labels=labels)

    data_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    val_dataset = datasets.ImageFolder(root=os.path.join(config.data_root, "val", config.snr[10]),
                                       transform=data_transforms)
    for i in range(10):
        val_ds = datasets.ImageFolder(root=os.path.join(config.data_root, "val", config.snr[i]),
                                      transform=data_transforms)
        val_dataset = ConcatDataset([val_dataset, val_ds])
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=True,
                                             num_workers=config.n_works)
    net = create_model(num_classes=config.classes)
    # 加载预训练权重
    config.weights_file = 'swin_tiny_patch4_windows7_224.pth'
    assert os.path.exists(config.weights_file), "cannot find {} file".format(config.weights_file)
    net.load_state_dict(torch.load(config.weights_file, map_location=config.device))
    net.to(config.device)
    net.eval()
    with torch.no_grad():
        for val_data in tqdm(val_loader, file=sys.stdout):
            val_images, val_labels = val_data
            outputs = net(val_images.to(config.device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()


if __name__ == '__main__':
    main()
