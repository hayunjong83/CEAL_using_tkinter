import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from typing import Tuple, Optional, Callable, Union
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, root_dir, labeled, class_label, transform=None):
        self.root_dir = os.path.expanduser(root_dir)
        self.class_label = class_label

        self.data = []
        self.labels = []
        self.class_label = class_label
        self._classes = len(self.class_label)
        self.transform = transform

        if labeled:
            for i, label in enumerate(class_label):
                label_dir = os.path.join(self.root_dir, label)
                files = os.listdir(label_dir)
                for file in files:
                    self.data.append(os.path.join(label_dir, file))
                    self.labels.append(i)
        else:
            files = os.listdir(self.root_dir)
            for file in files:
                self.data.append(os.path.join(root_dir, file))
                self.labels.append(-1)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx], self.labels[idx]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)
        
        sample = {'image': img, 'label': label, 'filename': img_path}
        return sample
    
    def __len__(self):
        return len(data)

        



class CatsAndDogs(Dataset):
    def __init__(self, root_dir: str = "../data/dl",
                 labeled : bool = True,
                 transform: Optional[Callable] = None):

        self.root_dir = os.path.expanduser(root_dir)
        self.transform = transform
        self.data = []
        self.labels = []
        self._classes = 2

        category = ['cats', 'dogs']

        if labeled :
            for cat in category:
                cat_dir = os.path.join(root_dir, cat)

                for img in glob.glob(os.path.join(cat_dir, '*.jpg')):
                    self.data.append(img)
                    if cat == 'cats':
                        self.labels.append(0)
                    elif cat == 'dogs':
                        self.labels.append(1)
        else:
            for img in glob.glob(os.path.join(root_dir,'*.jpg')):
                self.data.append(img)
                self.labels.append(-1)

    
    
    def __getitem__(self, idx: int) -> dict:
        img, label = self.data[idx], self.labels[idx]
        original_filename = img
        img = cv2.imread(img)
        img = img[:, :, ::-1]
        img = self.img_normalize(img)
        sample = {'image': img, 'label': label,
                    'filename': original_filename }

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def __len__(self):

        return len(self.data)

    @staticmethod
    def img_normalize(img):
        img = (img / 255.0)

        return img

class Normalize(object):
    def __init__(self, mean: np.ndarray = np.array([0.485, 0.456, 0.406]),
                 std: np.ndarray = np.array([0.229, 0.224, 0.225])):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img, label, filename = sample['image'], sample['label'], sample['filename']
        img = img - self.mean
        img /= self.std
        sample = {'image': img, 'label': label, 'filename': filename}
        return sample

class SquarifyImage(object):
    
    def __init__(self, box_size: int = 256, scale: tuple = (0.6, 1.2),
                 is_scale: bool = True,
                 seed: Optional[Union[Callable, int]] = None):
        super(SquarifyImage, self).__init__()
        self.box_size = box_size
        self.min_scale_ratio = scale[0]
        self.max_scale_ratio = scale[1]
        self.is_scale = is_scale
        self.seed = seed

    def __call__(self, sample):
        img, label, filename = sample['image'], sample['label'], sample['filename']
        img = self.squarify(img)
        sample = {'image': img, 'label': label, 'filename': filename}
        return sample

    def squarify(self, img):
        if self.is_scale:
            img_scaled = self.img_scale(img)
            img = img_scaled
        w, h, _ = img.shape

        ratio = min(self.box_size / w, self.box_size / h)
        resize_w, resize_h = int(w * ratio), int(h * ratio)
        x_pad, y_pad = (self.box_size - resize_w) // 2, (
                self.box_size - resize_h) // 2
        t_pad, b_pad = x_pad, self.box_size - resize_w - x_pad
        l_pad, r_pad = y_pad, self.box_size - resize_h - y_pad

        resized_img = cv2.resize(img, (resize_h, resize_w))

        img_padded = cv2.copyMakeBorder(resized_img,
                                        top=t_pad,
                                        bottom=b_pad,
                                        left=l_pad,
                                        right=r_pad,
                                        borderType=0,
                                        value=0)

        if img_padded.shape == [self.box_size, self.box_size, 3]:
            raise ValueError(
                'Invalid size for squarified image {} !'.format(
                    img_padded.shape))
        return img_padded

    def img_scale(self, img):
        scale = np.random.uniform(self.min_scale_ratio, self.max_scale_ratio,
                                  self.seed)
        img_scaled = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        return img_scaled

class RandomCrop(object):
    """
    Randomly crop the image in the sample to a target size
    target_size: tuple(int, int) or int. If int, take a square crop.
        the desired crop size
    """

    def __init__(self, target_size: Union[tuple, int]):

        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            assert len(target_size) == 2
            self.target_size = target_size

    def __call__(self, sample):

        img, label, filename = sample['image'], sample['label'], sample['filename']
        h, w = img.shape[:2]
        new_h, new_w = self.target_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = img[top: top + new_h, left: left + new_w]
        sample = {'image': img, 'label': label, 'filename': filename}
        return sample


class ToTensor(object):
    """
    Convert ndarrays image in sample to pytorch Tensor
    """

    def __call__(self, sample):
        img, label, filename = sample['image'], sample['label'], sample['filename']
        img = img.transpose((2, 0, 1))
        sample = {'image': torch.from_numpy(img),
                  'label': label,
                  'filename': filename}
        return sample