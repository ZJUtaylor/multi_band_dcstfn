import sys
import logging
import shutil
import csv
import os 
import torch

import rasterio
import warnings
from osgeo import gdal
gdal.PushErrorHandler('CPLQuietErrorHandler')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def make_tuple(x):
    if isinstance(x, int):
        return x, x
    if isinstance(x, list):
        if len(x) == 1:
            return x[0], x[0]
        else:
            return x[0], x[1]
    else:
        return x


def cov(x, y):
    return torch.mean((x - torch.mean(x)) * (y - torch.mean(y)))


def kge(prediction, target):
    m_true = torch.mean(target)
    m_pred = torch.mean(prediction)
    std_true = torch.std(target)
    std_pred = torch.std(prediction)
    r = cov(target, prediction) / (std_true * std_pred)
    return (1 - torch.sqrt((r - 1) ** 2
                           + (std_pred / std_true - 1) ** 2
                           + (m_pred / m_true - 1) ** 2))


def ssim(prediction, target, data_range=10000):
    K1 = 0.01
    K2 = 0.03
    L = data_range

    mu_x = prediction.mean()
    mu_y = target.mean()

    sig_x = prediction.std()
    sig_y = target.std()
    sig_xy = cov(target, prediction)

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    return ((2 * mu_x * mu_y + C1) * (2 * sig_xy + C2) /
            ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2)))


def score(prediction, target, metric):
    # print("in utils.score function, prediction.shape is ", prediction.shape)
    # print("in utils.score function, target.shape is ", target.shape)
    assert prediction.shape == target.shape
    prediction = prediction.detach() * 10000
    target = target.detach() * 10000

    if prediction.dim() == 2:
        return metric(prediction.view(-1), target.view(-1)).item()

    if prediction.dim() == 3:
        n_samples = prediction.shape[0]
        value = 0.0
        for i in range(n_samples):
            value += metric(prediction[i].view(-1), target[i].view(-1)).item()
        value = value / n_samples
        return value
    if prediction.dim() == 4: #
        value = 0.0
        n_samples = prediction.shape[0]
        n_channel = prediction.shape[1]
        for i in range(n_samples):
            for j in range(n_channel):
                value += metric(prediction[i][j].view(-1), target[i][j].view(-1)).item()
        value = value / (n_samples * n_channel)
        return value

    else:
        raise ValueError('The dimension of the inputs is not right.')


def save_array_as_tif(matrix, filepath, profile=None, prototype=None):
    if prototype:
        with rasterio.open(str(prototype)) as src:
            profile = src.profile
    if not profile:
        warnings.warn('the geographic profile is not provided')
    with rasterio.open(filepath, mode='w', **profile) as dst:
        dst.write(matrix)


def get_logger(logpath=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # 文件日志
        if logpath:
            file_handler = logging.FileHandler(logpath)
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(file_handler)

        # 控制台日志
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
    return logger


def save_checkpoint(state, is_best, checkpoint='last.pth', best='best.pth'):
    torch.save(state, checkpoint)

    if is_best:
        shutil.copy(str(checkpoint), str(best))


def load_checkpoint(checkpoint, model, optimizer=None):
    # if not checkpoint.exists():
    #     raise FileNotFoundError(f"File doesn't exist {checkpoint}")
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def log_csv(filepath, values, header=None, multirows=False):
    empty = False
    # if not os.path.exists(filepath):
    #     filepath.touch()
    #     empty = True

    with open(filepath, 'a') as file:
        writer = csv.writer(file)
        if empty and header:
            writer.writerow(header)
        if multirows:
            writer.writerows(values)
        else:
            writer.writerow(values)

