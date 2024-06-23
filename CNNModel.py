import os
import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, precision_score, recall_score, f1_score
from PIL import Image
import matplotlib.pyplot as plt
import platform
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, Sequential, Conv2d, LeakyReLU, BatchNorm2d, MaxPool2d, Dropout, Linear, ReLU, CrossEntropyLoss


class CNNModel(Module):
    def __init__(self, no_classes):
        super(CNNModel, self).__init__()
        self.layer1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(32),
            LeakyReLU(inplace=True),
            Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(32),
            LeakyReLU(inplace=True),
            MaxPool2d(stride=2, kernel_size=2)
        )
        self.layer2 = Sequential(
            Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            LeakyReLU(inplace=True),
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            LeakyReLU(inplace=True),
            MaxPool2d(stride=2, kernel_size=2)
        )
        self.fc1 = Sequential(
            Dropout(p=0.1),
            Linear(64*16*16, 1000),
            ReLU(inplace=True),
            Linear(1000, 512),
            ReLU(inplace=True),
            Dropout(p=0.1),
            Linear(512, no_classes)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out