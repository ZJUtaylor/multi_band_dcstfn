import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import os
from model import FusionNet
from data import PatchSet, get_image_pair
import utils

from timeit import default_timer as timer
from datetime import datetime
import numpy as np
import pandas as pd
import shutil
from osgeo import gdal
gdal.PushErrorHandler('CPLQuietErrorHandler')

class Experiment(object):
    def __init__(self, option):
        # self.device = torch.device('cuda' if option.cuda else 'cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scale = 1
        self.image_size = option.image_size
        self.option = option

        self.save_dir = option.save_dir
        # if os.path.exists(self.save_dir):
        #     shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.train_dir = os.path.join(self.save_dir, 'train')
        os.makedirs(self.train_dir, exist_ok=True)
        self.train_log = os.path.join(self.save_dir, 'train', '%s_train_log.txt' % option.dataset)

        self.history = os.path.join(self.train_dir, 'history.csv')
        self.test_dir = os.path.join(self.save_dir, 'test')
        os.makedirs(self.test_dir, exist_ok=True)
        self.checkpoint = os.path.join(self.train_dir, 'last.pth')

        self.best = os.path.join(self.train_dir, 'best.pth')
        self.logger = utils.get_logger()
        self.logger.info('Model initialization')

        self.model = FusionNet(option.input_channel, option.channels).to(self.device)
        if option.ngpu > 1:
            self.model = nn.DataParallel(self.model,
                                         device_ids=[i for i in range(option.ngpu)])

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=option.lr)

        self.logger.info(str(self.model))
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f'There are {n_params} trainable parameters.')

    def train_on_epoch(self, n_epoch, train_loader):
        # epoch_loss = utils.AverageMeter()
        # epoch_score = utils.AverageMeter()
        epoch_loss, epoch_ssim = [], []
        batches = len(train_loader)
        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')
        t_start = timer()
        for idx, (inputs, target) in enumerate(train_loader):
            
            inputs = [im.to(self.device) for im in inputs]
            target = target.to(self.device)

            self.optimizer.zero_grad()
            prediction = self.model(inputs[-1], inputs[:-1])
            # print("prediction shape is ", prediction.shape) #  torch.Size([32, 6, 210, 196])
            # print("target shape is ", target.shape)
            loss = self.criterion(prediction, target)
            # epoch_loss.update(loss.item())
            epoch_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()
            # print("*" * 20, "debug", "*" * 20)
            # print("predictoion shape is ", prediction.shape)  # target shape is  torch.Size([32, 1, 80, 80])
            # print("target shape is ", target.shape)
            # ssim_epoch.append(utils.ssim(prediction, target))
            score_ssim = utils.score(prediction, target, utils.ssim)
            t_end = timer()
            print(f"Epoch: {n_epoch}, {idx} / {batches}, cost: {t_end - t_start}'s, ssim: {score_ssim}")
            t_start = t_end
            # score_kge = utils.score(prediction, target, utils.kge)
            # epoch_score.update(score)
            epoch_ssim.append(score_ssim)
            

            # self.logger.info(f'Epoch[{n_epoch} {idx}/{batches}] - '
            #                  f'Loss: {loss.item():.10f} - '
            #                  f'SSIM: {score:.5f} - '
            #                  f'Time: {t_end - t_start}s')
            # with open(self.train_log, 'a') as f:
            #     f.write(log_str + '\n')

        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')
        return np.array(epoch_loss).mean(), np.array(epoch_loss).mean()

    def test_on_epoch(self, n_epoch, val_loader, best_acc):
        epoch_loss = utils.AverageMeter()
        epoch_score = utils.AverageMeter()
        with torch.no_grad():
            for inputs, target in val_loader:
                inputs = [im.to(self.device) for im in inputs]
                target = target.to(self.device)
                prediction = self.model(inputs[-1], inputs[:-1])
                loss = self.criterion(prediction, target)
                epoch_loss.update(loss.item())
                score = utils.score(prediction, target, utils.ssim)
                epoch_score.update(score)
                
        # with open(self.train_log, 'a') as f:
        #     log_str = 'test: ' + str(epoch_score.avg)
        #     f.write(log_str + '\n')
        # 记录Checkpoint
        is_best = epoch_score.avg >= best_acc
        state = {'epoch': n_epoch,
                 'state_dict': self.model.state_dict(),
                 'optim_dict': self.optimizer.state_dict()}
        utils.save_checkpoint(state, is_best=is_best,
                              checkpoint=self.checkpoint,
                              best=self.best)
        return epoch_loss.avg, epoch_score.avg

    def train(self, train_dir, val_dir, patch_size, patch_stride, batch_size, num_workers=10, epochs=30, resume=True):
        # 加载数据
        self.logger.info('Loading data...')
        train_set = PatchSet(train_dir, self.image_size, patch_size, patch_stride, scale=self.scale)
        print("finish train set init....")
        val_set = PatchSet(val_dir, self.image_size, patch_size, scale=self.scale)
        print("finish val set init")
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=num_workers, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, drop_last=True)

        best_val_acc = 0.0
        start_epoch = 0

        # if resume and self.checkpoint.exists():
        #     utils.load_checkpoint(self.checkpoint, model=self.model, optimizer=self.optimizer)
        #     if self.history.exists():
        #         df = pd.read_csv(self.history)
        #         best_val_acc = df['val_acc'].max()
        #         start_epoch = int(df.iloc[-1]['epoch']) + 1

        self.logger.info('Training...')
        scheduler = ExponentialLR(self.optimizer, 0.75, last_epoch=start_epoch - 1)
        for epoch in range(start_epoch, epochs + start_epoch):
            scheduler.step()
            # 输出学习率
            for param_group in self.optimizer.param_groups:
                self.logger.info(f"Current learning rate: {param_group['lr']}")

            train_loss, train_score = self.train_on_epoch(epoch, train_loader)
            val_loss, val_score = self.test_on_epoch(epoch, val_loader, best_val_acc)
            with open('./save_dir/train_log.txt', 'a') as f:
                str_log = 'Epoch: ' + str(epoch) + '\ntrain: ' + "train_loss: " + str(train_loss) + "train ssim: " + str(train_score)\
                    +'\ntest: ' + "test_loss: " + str(val_loss) + "test ssim: " + str(val_score) + '\n'
                f.write(str_log)
            csv_header = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
            csv_values = [epoch, train_loss, train_score, val_loss, val_score]
            utils.log_csv(self.history, csv_values, header=csv_header)
            # self.test(self.val_dir,self.patch_size)

    def test(self, test_dir, patch_size, num_workers=0):
        self.model.eval()
        patch_size = utils.make_tuple(patch_size)
        utils.load_checkpoint(self.best, model=self.model)
        self.logger.info('Testing...')
        # 记录测试文件夹中的文件路径，用于最后投影信息的匹配
        # image_dirs = [p for p in test_dir.glob('*') if p.is_dir()]
        image_dirs = [f for f in os.listdir(test_dir) if '.DS_Store' not in f]
        image_dirs = sorted([os.path.join(test_dir, g) for g in image_dirs])
        print("image dirs in test is", image_dirs)
        image_paths = [get_image_pair(d) for d in image_dirs]

        # 在预测阶段，对图像进行切块的时候必须刚好裁切完全，这样才能在预测结束后进行完整的拼接
        assert self.image_size[0] % patch_size[0] == 0
        assert self.image_size[1] % patch_size[1] == 0
        rows = int(self.image_size[1] / patch_size[1])
        cols = int(self.image_size[0] / patch_size[0])
        n_blocks = rows * cols  # 一张图像中的分块数目
        test_set = PatchSet(test_dir, self.image_size, patch_size, scale=self.scale)
        test_loader = DataLoader(test_set, batch_size=1, num_workers=num_workers)

        scaled_patch_size = tuple([i * self.scale for i in patch_size])
        scaled_image_size = tuple([i * self.scale for i in self.image_size])
        
        scale_factor = 10000
        with torch.no_grad():
            im_count = 0
            patches = []
            t_start = datetime.now()
            for inputs, _ in test_loader:
                name = image_paths[im_count][2]
                if len(patches) == 0:
                    t_start = timer()
                    self.logger.info(f'Predict on image {name}')

                # 分块进行预测（每次进入深度网络的都是影像中的一块）
                inputs = [im.to(self.device) for im in inputs]
                
                # print("debug, inputs shape is ", inputs[0].shape)
                prediction = self.model(inputs[-1], inputs[:-1])
                # prediction = prediction.cpu().numpy().reshape(scaled_patch_size)
                prediction = prediction.squeeze(0).cpu().numpy()
                image_bands = int(prediction.shape[0])
                # print("prediction shape is ", prediction.shape) # (6, 40, 40)
                patches.append(prediction * scale_factor)

                # 完成一张影像以后进行拼接
                if len(patches) == n_blocks:
                    result = np.empty((image_bands, scaled_image_size[0], scaled_image_size[1]), dtype=np.float32)
                    # print("result shape is ", result.shape)
                    block_count = 0
                    for i in range(rows):
                        row_start = i * scaled_patch_size[1]
                        for j in range(cols):
                            col_start = j * scaled_patch_size[0]
                            result[:,
                                row_start: row_start + scaled_patch_size[1],
                                col_start: col_start + scaled_patch_size[0]
                            ] = patches[block_count]
                            block_count += 1
                    patches.clear()
                    # 存储预测影像结果
                    result = result.astype(np.int16)
                    prototype = str(image_paths[im_count][1])

                    os.makedirs('./prediction_dcstfn/%s' % self.option.dataset, exist_ok=True)
                    # No such file or directory: './prediction_dcstfn/Daxing_dataset/dcstfn_../public_data/Daxing_dataset/test_demo/group_000/L_2019-11-05.tif.npy'
                    save_name = image_paths[im_count][3].split("/")
                    save_name = save_name[-2] + '-' + save_name[-1].replace(".tif", "")
                    np.save(os.path.join('./prediction_dcstfn', self.option.dataset, "dcstfn_%s" % save_name), result)
                    
                    # utils.save_array_as_tif(result, self.test_dir / name, prototype=prototype)
                    im_count += 1
                    t_end = timer()
                    self.logger.info(f'Time cost: {t_end - t_start}s')
