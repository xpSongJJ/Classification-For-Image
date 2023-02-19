import math
import os
import torch.nn as nn
import torch


class Config:
    def __init__(self):
        self.epochs = 5
        self.batch_size = 24
        self.data_root = 'G:/SXP/data/image'
        self.snr = ['0dB_tf', '2dB_tf', '-2dB_tf', '4dB_tf', '-4dB_tf',
                    '6dB_tf', '-6dB_tf', '8dB_tf', '-8dB_tf', '10dB_tf', '-10dB_tf']
        self.classes = 12
        self.n_works = min([os.cpu_count(), self.batch_size if self.batch_size > 1 else 0, 8])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_name = "vgg16"
        self.loss_function = nn.CrossEntropyLoss()
        self.learning_rate = 0.0002
        self.alpha = 0.01
        self.cosine_decay = lambda x: ((1 + math.cos(x * math.pi / self.epochs)) / 2) * (1 - self.alpha) + self.alpha
        self.weights_file = "weights.pth"
        self.num_train = 0
        self.num_val = 0


