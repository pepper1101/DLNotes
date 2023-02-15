import  math
from typing import Generic, List, Tuple
from functools import reduce

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import T_co

_IMG_N_COL = 4

def show_images(imgs):
    '''
    打印多张图片，每行多张最多4张图片，要求保持图片比例不变
    '''

    # 计算图片占用行/列数
    num_imgs = len(imgs)
    num_cols = min(num_imgs, 4)
    num_rows = math.ceil(num_imgs / num_cols)

    # 计算所有图片最大宽和最大高。
    # 最大高，即最大的y坐标
    max_width = reduce(lambda x, y : x if x.size[0] > y.size[0] else y, imgs).size[0]
    max_height = reduce(lambda x, y : x if x.size[1] > y.size[1] else y, imgs).size[1]

    # dpi 即每英寸像素数
    dpi = float(plt.rcParams['figure.dpi'])
    figsize = max_width / dpi * num_cols, max_height / dpi * num_rows

    # sharey 是关键，共享y坐标，保证图片的尺寸比例不变
    # 不同列之间sharey，不同行之间sharex
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=False, sharey=True)
    # plt.xlim(0, math.ceil(max_width / 100) * 100)
    plt.ylim(math.ceil(max_height / 100) * 100, 0)

    axes = axes.reshape(-1)
    for n in range(num_imgs):
        i, j = n // num_cols, n % num_cols
        axis_img = axes[n].imshow(imgs[n], cmap='gray')
        axes[n].axes.get_xaxis().set_visible(True)
        axes[n].axes.get_yaxis().set_visible(True)


def _show_PIL_img(img, label):
    '''
    '''
    ax = plt.subplot(111)
    ax.set_title(label)
    ax.imshow(img)
    plt.show()

def _show_PIL_imgs(imgs, labels):
    '''
    BCHW
    '''
    if len(imgs) == 1:
        _show_PIL_img(imgs, labels)
    else:
        nrows = math.ceil(float(imgs.shape[0]) /_IMG_N_COL)
        ncols = min(_IMG_N_COL, imgs.shape[0])
        print(nrows, ',', ncols)
        _, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True)
        axes = axes.reshape(-1)

        for i in range(imgs.shape[0]):
            axes[i].set_title(labels[i])
            axes[i].imshow(imgs[i].permute(1, 2, 0))


def _show_tensor_img(img, label):
    '''
    CHW
    '''
    ax = plt.subplot(111)
    ax.set_title(label)
    ax.imshow(img.permute(1, 2, 0))
    plt.show()

def _show_tensor_imgs(imgs, labels):
    '''
    BCHW
    '''
    if len(imgs.shape) == 3:
        _show_tensor_img(imgs, labels)
    else:
        nrows = math.ceil(float(imgs.shape[0]) / _IMG_N_COL)
        ncols = min(_IMG_N_COL, imgs.shape[0])
        _, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True)
        axes = axes.reshape(-1)

        for i in range(imgs.shape[0]):
            axes[i].set_title(labels[i])
            axes[i].imshow(imgs[i].permute(1, 2, 0))

class ImageAugumentsLoader(Generic[T_co]):
    '''
    应用多个图像增广转换（Transform），一张图片变换成多张图片，参与机器学习训练，有效降低过拟合
    ComposeAugumentsLoader内嵌一个DataLoader，将内嵌的DataLoader输出的每张图片转换成多张图片输出
    应用的每个图像增广转换（Transform）对应一个执行次数multi，每次执行会生成随机不同的结果
    '''

    _inner_loader: DataLoader[T_co]
    _trans_multi: List[Tuple[nn.Module, int]]

    def __init__(self, loader: DataLoader[T_co]) -> None:
        self._inner_loader = loader
        self._trans_multi  = []

    def add_trans(self, trans, multi = 8) -> 'ImageAugumentsLoader':
        self._trans_multi.append((trans, multi))
        return self
    
    def sum_multi(self):
        return sum(ele[1] for ele in self._trans_multi)

    def __len__(self):
        return (1 + self.sum_multi()) * len(self._inner_loader)
    
    # for iterator

    _multi: int
    _inner_iter: object
    _i: int
    
    def __iter__(self) -> 'ImageAugumentsLoader':
        self._multi = self.sum_multi()
        self._inner_iter = iter(self._inner_loader)
        self._i = -1

        return self
    
    def _transf(self):
        imgs, labels = self._batch
        imgs_, labels_ = imgs, labels

        for trans, multi in self._trans_multi:
            for i in range(multi):
                imgs_ = torch.cat((imgs_, trans(imgs)))
                labels_ = torch.cat((labels_, labels))
        
        return imgs_, labels_
    
    def __next__(self):
        self._batch = self._inner_iter.__next__()
        return self._transf()
