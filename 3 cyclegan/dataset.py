import os
import torch
import cv2
from PIL import Image
from sympy.vector import express
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import Resize, ToTensor, Compose
import pandas as pd
import numpy as np

import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class Affectnet(Dataset):
    def __init__(self, root, is_train=True, transform=None):
        self.image_path = os.path.join(root, "Manually_Annotated_Images")
        self.transform = transform

        label_path = os.path.join(root, "training.csv" if is_train else "validation.csv")
        list_label = pd.read_csv(label_path)

        self.neutral_images = list_label[list_label['expression'] == 0]['subDirectory_filePath'].tolist()
        self.happy_images = list_label[list_label['expression'] == 1]['subDirectory_filePath'].tolist()

        self.neutral_len = len(self.neutral_images)
        self.happy_len = len(self.happy_images)
        self.length_dataset = max(self.neutral_len, self.happy_len)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        try:
            neutral_img_path = os.path.join(self.image_path, self.neutral_images[index % self.neutral_len])
            happy_img_path = os.path.join(self.image_path, self.happy_images[index % self.happy_len])

            neutral_img = Image.open(neutral_img_path).convert("RGB")
            happy_img = Image.open(happy_img_path).convert("RGB")

            if self.transform:
                neutral_img = self.transform(neutral_img)
                happy_img = self.transform(happy_img)

            return neutral_img, happy_img

        except (FileNotFoundError, OSError) as e:
            print(f"Error loading images: {e}")
            next_index = (index + 1) % self.length_dataset
            return self.__getitem__(next_index)

    def cout(self, a):
        print(a)
        print(type(a))
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = Compose([
        ToTensor(),
        Resize((224, 224))
    ])

    train_dataset = Affectnet(root="C:/Users/tam/Documents/Data/Affectnet", is_train=True,
                              transform=transform)
    print(train_dataset.__len__())
    a = train_dataset[0]
