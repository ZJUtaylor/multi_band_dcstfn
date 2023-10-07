import argparse
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os 
import data
from experiment import Experiment
import rasterio
from osgeo import gdal
gdal.PushErrorHandler('CPLQuietErrorHandler')
# 获取模型运行时必须的一些参数
parser = argparse.ArgumentParser(description='Acquire some parameters for fusion model')
parser.add_argument('--dataset', type=str, default='AHB_dataset', choices=['AHB_dataset', 'Daxing_dataset', 'Tianjin_dataset', 'Private_dataset'], help='the test data directory')
parser.add_argument('--channels', type=int, nargs='+', default=[32, 32, 32],
                    help='the numbers of features in each abstract level')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='the initial learning rate')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=2,
                    help='number of epochs to train')
# parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--num_workers', type=int, default=4, help='number of threads to load data')
# parser.add_argument('--save_dir', type=str, default='./save_dir', help='the output directory')


# 获取对输入数据进行预处理时的一些参数


opt = parser.parse_args()
opt.train_dir = '/home/taylor/Desktop/yxt/homework/Data/%s/train' % opt.dataset
opt.test_dir = '/home/taylor/Desktop/yxt/homework/Data/%s/test' % opt.dataset
opt.save_dir = './save_dir/%s' % opt.dataset

opt.image_size = []
train_instance = os.path.join(opt.train_dir + '/group_000', os.listdir(opt.train_dir + '/group_000')[0])
with rasterio.open(str(train_instance)) as ds:
    im = ds.read().astype(np.float32)   
    opt.image_size = (im.shape[1], im.shape[2])
    opt.input_channel = im.shape[0]  # single band 

if opt.dataset == 'Tianjin_dataset':
    opt.patch_size = (10, 10)
elif opt.dataset == 'AHB_dataset': # (2480, 2800)
    opt.patch_size = (80, 80)
else:
    opt.patch_size = (40, 40)

print("opt image size is ", opt.image_size)

assert opt.image_size[0] % opt.patch_size[0] == 0 and opt.image_size[1] % opt.patch_size[1] == 0
opt.patch_stride = opt.patch_size  # todo 
opt.test_patch = opt.patch_size


if __name__ == '__main__':
    # important: band 
    # if opt.cuda and not torch.cuda.is_available():
    #     opt.cuda = False
    # else:
    cudnn.benchmark = True
    cudnn.deterministic = True
    experiment = Experiment(opt)
    if opt.epochs > 0:
        experiment.train(opt.train_dir, opt.test_dir, opt.patch_size, opt.patch_stride, opt.batch_size, num_workers=opt.num_workers, epochs=opt.epochs)
    experiment.test(opt.test_dir, opt.test_patch, num_workers=opt.num_workers)
