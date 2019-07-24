import os
import torch.utils.data as data
import cv2

class ImageDataset(data.Dataset):
    """
    Args:
        dataset_dir: directory of dataset
        transform: dataset transform
    """
    def  __init__(self, dataset_dir,  transform=None):
        self.image_info = []
        self.transform = transform
        images = next(os.walk(dataset_dir))[2]
        for img in images:
            self.image_info.append(os.path.join(dataset_dir, img))
    
    def __len__(self):
        return len(self.image_info)

    def  __getitem__(self, index):
        image = cv2.imread(self.image_info[index])
        #将BGR转化为RGB通道顺序
        image=image[:,:,::-1]
        if self.transform:
                image = self.transform(image)
        return image