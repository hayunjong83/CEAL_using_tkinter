from typing import Tuple, Optional, Callable, Union
from torch.utils.data import Dataset

import torch
import numpy as np
import os
import glob
import cv2

def least_confidence(pred_prob: np.ndarray, 
                    k: int) -> Tuple[np.ndarray, np.ndarray]:
    most_pred_prob, most_pred_class = np.max(pred_prob, axis=1), np.argmax(pred_prob, axis=1)
    size = len(pred_prob)
    lc_i = np.column_stack((list(range(size)), most_pred_class, most_pred_prob))
    lc_i = lc_i[lc_i[:, -1].argsort()]

    return lc_i[:k, 0].astype(np.int32), lc_i[:k]

def margin_sampling(pred_prob: np.ndarray, k: int) -> Tuple[np.ndarray,
                                                            np.ndarray]:
    size = len(pred_prob)
    margin = np.diff(np.abs(np.sort(pred_prob, axis=1)[:, ::-1][:, :2]))
    pred_class = np.argmax(pred_prob, axis=1)
    ms_i = np.column_stack((list(range(size)), pred_class, margin))

    # sort ms_i in ascending order according to margin
    ms_i = ms_i[ms_i[:, 2].argsort()]

    # the smaller the margin  means the classifier is more
    # uncertain about the sample
    return ms_i[:k, 0].astype(np.int32), ms_i[:k]


def entropy(pred_prob: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:

    size = len(pred_prob)
    #entropy_ = - np.nansum(pred_prob * np.log(pred_prob), axis=1)
    entropy_ = - np.nansum( np.exp(pred_prob) * pred_prob, axis=1)
    pred_class = np.argmax(pred_prob, axis=1)
    en_i = np.column_stack((list(range(size)), pred_class, entropy_))

    # Sort en_i in descending order
    en_i = en_i[(-1 * en_i[:, 2]).argsort()]
    return en_i[:k, 0].astype(np.int32), en_i[:k]

def get_uncertain_samples(pred_prob: np.ndarray, k: int,
                          criteria: str) -> Tuple[np.ndarray, np.ndarray]:
    
    if criteria == 'cl':
        uncertain_samples = least_confidence(pred_prob=pred_prob, k=k)
    elif criteria == 'ms':
        uncertain_samples = margin_sampling(pred_prob=pred_prob, k=k)
    elif criteria == 'en':
        uncertain_samples = entropy(pred_prob=pred_prob, k=k)
    else:
        raise ValueError('criteria {} not found !'.format(criteria))
    return uncertain_samples

def get_high_confidence_samples(pred_prob: np.ndarray,
                                delta: float) -> Tuple[np.ndarray, np.ndarray]:
    _, eni = entropy(pred_prob=pred_prob, k=len(pred_prob))
    print(eni)
    hcs = eni[eni[:, 2] < delta]
    print("-"*10)
    print(hcs)

    return hcs[:, 0].astype(np.int32), hcs[:, 1].astype(np.int32)

def update_threshold(delta: float, dr: float, t: int) -> float:
    if t>0:
        delta = delta - dr * t
    return delta

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