import os
import cv2
import numpy as np
import torch.utils.data as data
import random


class ImageDataset(data.Dataset):
    """
    Args:
        dataset_dir: directory of dataset
        transform: dataset transform
    """

    def __init__(self, dataset_dir, sorted=False,  transform=None):
        # 随机取1000张图片
        self.image_info = []
        self.transform = transform
        images = next(os.walk(dataset_dir))[2]
        if sorted:
            images.sort()
        for img in images:
            if img.endswith('.png') or img.endswith('.jpg') or img.endswith('.jpeg'):
                self.image_info.append(os.path.join(dataset_dir, img))

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, index):
        image = cv2.imread(self.image_info[index])
        # 将BGR转化为RGB通道顺序
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image
