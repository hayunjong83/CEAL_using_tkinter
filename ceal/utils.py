from typing import Tuple, Optional, Callable, Union

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

