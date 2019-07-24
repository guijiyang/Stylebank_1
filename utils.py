import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
import random
import args
from time import sleep


def showimg(img):
    """
    Input a pytorch image tensor with size (channel, width, height) and display it.
    """
    img = img.clamp(min=0, max=1)
    img = img.cpu().numpy().transpose(1, 2, 0)
    plt.imshow(img)
    plt.show()


class Resize(object):
    """
    Resize with aspect ration preserved.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        m = min(img.shape[0:2])
        new_size = (int(img.shape[1] / m * self.size),int(img.shape[0] / m * self.size))
        return cv2.resize(img, new_size)


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(img, output_size):
        h, w = img.shape[0:2]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.size)
        return img[i:i+h, j:j+w]


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        h, w = img.shape[0:2]
        th, tw = self.size

        if w == tw and h == th:
            return img[0:h,0:w]

        i = (h-th)//2
        j = (w-tw)//2
        return img[i:i+th,j:j+tw]


def adjust_learning_rate(optimizer, step):
    """
    Learning rate decay
    """
    lr = max(args.lr * (0.8 ** (step)), 1e-6)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_sid_batch(style_id_seg, batch_size):
    ret = style_id_seg
    while len(ret) < batch_size:
        ret += style_id_seg

    return ret[:batch_size]


content_img_transform = transforms.Compose([
    Resize(513),
    RandomCrop([513, 513]),
    transforms.ToTensor(),
])

style_img_transform = transforms.Compose([
    Resize(513),
    CenterCrop([513, 513]),
    transforms.ToTensor(),
])
