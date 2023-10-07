from pathlib import Path
import numpy as np
import rasterio
import os
from PIL import Image
import math
from collections import OrderedDict

import torch
from torch.utils.data import Dataset
from utils import make_tuple





def load_image_pair(im_dir, scale):
    """
    从指定目录中加载一组高低分辨率的图像对
    """
    # 按照一定顺序获取给定文件夹下的一组数据
    paths = get_image_pair(im_dir)

    # 将组织好的数据转为Image对象
    images = []
    for p in paths:
        # with rasterio.open(str(p)) as ds:
        ims = rasterio.open(str(p)).read().astype(np.float32)
        ims = np.array(ims)
        one_image = []
        for i in range(ims.shape[0]):
            im = Image.fromarray(ims[i])         # HW
            one_image.append(im)
        # im = Image.fromarray(im)      #todo: chanage for multi-channel  
        # print("in load_image_pair, im.shape is ", im.size)
        images.append(one_image)

    # 对数据的尺寸进行验证（等于4的时候是训练数据集，等于3的时候是实际预测）
    assert len(images) == 4 or len(images) == 3
    # 返回训练数据和验证数据（最后一个是验证数据）
    return images


def get_image_pair(im_dir):
    # 在该实验中，所有的图像都按照如下顺序进行组织
    # order = OrderedDict()
    # order[0] = reference + '_' + coarse_prefix  -> M_ref
    # order[1] = reference + '_' + fine_prefix    -> L_ref
    # order[2] = predict_prefix + '_' + coarse_prefix   ->M_pre
    # order[3] = predict_prefix + '_' + fine_prefix     -> L_pre
    # paths用于存储组织好的数据对

    files = [f for f in sorted(os.listdir(im_dir)) if '.DS_Store' not in f]  # [L_reference, L_predict, M_reference, M_predict]
    # print("files is ", files)
    
    paths = [None] * 4
    paths[0] = files[2]
    paths[1] = files[0]
    paths[2] = files[3]
    paths[3] = files[1]

    # print("show paths is \n", paths)
    paths = [os.path.join(im_dir, file) for file in paths]
    return paths


class PatchSet(Dataset):
    """
    每张图片分割成小块进行加载
    参考影像个数只支持1和2
    """

    def __init__(self, image_dir, image_size, patch_size, patch_stride=None, scale=1):
        # image_dir = ../pubilc_data/ABH dataset/train
        super(PatchSet, self).__init__()

        patch_size = make_tuple(patch_size)
        if patch_stride is None:
            patch_stride = patch_size
        else:
            patch_stride = make_tuple(patch_stride)

        self.root_dir = image_dir
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.scale = scale

        # self.image_dirs = [p for p in self.root_dir.glob('*') if p.is_dir()]
        self.image_dirs = [f for f in os.listdir(self.root_dir) if '.DS_Store' not in f]
        self.image_dirs = sorted([os.path.join(self.root_dir, g) for g in self.image_dirs])
        # print("image_dirs is ", self.image_dirs)
        self.num_im_pairs = len(self.image_dirs)

        # 计算出图像进行分块以后的patches的数目
        self.num_patches_x = math.ceil((image_size[0] - patch_size[0] + 1) / patch_stride[0])
        self.num_patches_y = math.ceil((image_size[1] - patch_size[1] + 1) / patch_stride[1])
        self.num_patches = self.num_im_pairs * self.num_patches_x * self.num_patches_y

    @staticmethod
    def im2tensor(im):
        im = [np.array(i, np.float32, copy=False) * 0.0001 for i in im] # (C, W, H)
        im = np.array(im)
        im = torch.from_numpy(im)
        im = im.transpose(1, 2).contiguous() # from CWH to CHW
        return im

    def __getitem__(self, index):
        id_n, id_x, id_y = self.map_index(index)
        images = load_image_pair(self.image_dirs[id_n], self.scale)
        patches = [None] * len(images)

        for i in range(len(patches)):
            im = []
            for j in range(len(images[i])):
                im_j = images[i][j].crop([id_x, id_y,
                                     id_x + self.patch_size[0],
                                     id_y + self.patch_size[1]])
                im.append(np.array(im_j))
            patches[i] = self.im2tensor(im)

        del images[:]
        del images

        if len(patches) == 3:
            return patches, None
        return patches[:-1], patches[-1]

    def __len__(self):
        return self.num_patches

    def map_index(self, index):
        # 将全局的index映射到具体的图像对文件夹索引(id_n)，图像裁剪的列号与行号(id_x, id_y)
        id_n = index // (self.num_patches_x * self.num_patches_y)
        residual = index % (self.num_patches_x * self.num_patches_y)
        id_x = self.patch_stride[0] * (residual % self.num_patches_x)
        id_y = self.patch_stride[1] * (residual // self.num_patches_x)
        return id_n, id_x, id_y


# a = PatchSet("/home/taylor/Desktop/yxt/homework/Data/AHB_dataset/train", [2480, 2800], [10, 10], patch_stride=None, scale=1)
# x, y = a.__getitem__(0)
# for one in x:
#     print(one.shape)
# print(y.shape)