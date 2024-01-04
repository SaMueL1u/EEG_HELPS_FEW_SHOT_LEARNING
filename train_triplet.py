import argparse
import datetime
import random
from tensorboardX import SummaryWriter
import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.backends.cudnn as cudnn
from dataset.Few_dataset import Few_dataset
from model.img_model import img_extract
from model.DenseNet_264 import DenseNet
from model.Inception import Inception3
from model.ResNet50 import ResNet50
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn import Parameter

class Triplet(nn.Module):
    def __init__(self):
        super(Triplet, self).__init__()
        self.margin = 0.2

    def forward(self, anchor, positive, negative):
        pos_dist = (anchor - positive).pow(2).sum(1)
        neg_dist = (anchor - negative).pow(2).sum(1)
        loss = F.relu(pos_dist - neg_dist + self.margin)

        corr = 0
        batch_size = pos_dist.shape[0]
        for i in range(pos_dist.shape[0]):
            if pos_dist[i] > neg_dist[i]:
                corr += 1

        return loss.mean(), float(corr/batch_size)

def Triplet_loss(p1, q1):
    corr = 1
    margin = 0.2
    for i in range(1,q1.shape[0]):
        if (p1 - q1[i]).pow(2).sum(1) < (p1 - q1[i]).pow(2).sum(1):
            corr = 0
            pos_dist = (p1 - q1[0]).pow(2).sum(1)
            neg_dist = (p1 - q1[i]).pow(2).sum(1)
            loss = F.relu(pos_dist - neg_dist + margin)

    return loss.mean(), corr

epoch = 0
epoch = 6000
correct_train = 0
train_loss = 0
train_correct = 0
batch = 64
way = 10
shot = 1
correct_best = 0
snapshot_dir = './exp/few_shot/'

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

Path_img = './exp/use_pretrain/time_encoder_img14000_correct_Inception78.pth'
# Path_img = './exp/use_pretrain/flu_encoder_img17000_correct_Inception74.pth'
# Path_time_img = './exp/use_pretrain/time_encoder_img19000_correct_ResNet80.pth'
# Path_flu_img = './exp/use_pretrain/flu_encoder_img15000_correct_ResNet78.pth'

pre_eeg_true = False
model_name = 'Inception'
if pre_eeg_true:
    if model_name == 'ResNet':
        model = ResNet50(num_classes=1000)

        model.load_state_dict(torch.load(Path_img), strict=False)

        model.train()
        model = model.cuda()

    elif model_name == 'Inception':
        model = Inception3(num_classes=1000)

        model.load_state_dict(torch.load(Path_img), strict=False)

        model.train()
        model = model.cuda()

    elif model_name == 'DenseNet':
        model = DenseNet(num_classes=1000)

        model.load_state_dict(torch.load(Path_img), strict=False)

        model.train()
        model = model.cuda()

else:
    if model_name == 'ResNet':
        model = ResNet50(num_classes=1000).cuda()
    elif model_name == 'Inception':
        model = Inception3(num_classes=1000).cuda()
    elif model_name == 'DenseNet':
        model = DenseNet(num_classes=1000).cuda()



dataset = 'cars196'
if dataset == 'ImageNet':
    dataloader = Few_dataset(dataset=dataset, img_path=r'/local/ImageNet_Raw/train')
    Loss = Triplet()

elif dataset == 'cub':
    dataloader = Few_dataset(dataset=dataset, img_path=r'/local/liuyuntao/egg_channel_3/data/cub')
    Loss = Triplet()

elif dataset == 'cars196':
    dataloader = Few_dataset(dataset=dataset, img_path=r'/local/liuyuntao/egg_channel_3/data/cars196')
    Loss = Triplet()

if pre_eeg_true:
    writer = SummaryWriter('{}_{}_triplet_true_{}/scalar'.format(dataset, model_name, str(datetime.datetime.now().date())))
else:
    writer = SummaryWriter('{}_{}_triplet_false_{}/scalar'.format(dataset, model_name), str(datetime.datetime.now().date()))

if pre_eeg_true:
    param_groups = []
    param_groups.append({'params': list(set(model.parameters()).difference(set(model.fc.parameters()))), 'lr': 1e-4})
    param_groups.append({'params': model.fc.parameters(), 'lr': 1e-4})

    opt = optim.Adam(param_groups,weight_decay=5e-4)

else:
    param_groups = [{'params': list(set(model.parameters()).difference(set(model.fc.parameters()))), 'lr': 1e-4},
                      {'params': model.fc.parameters(), 'lr': 1e-4}]
    param_groups.append({'params': Loss.parameters()})
    opt = optim.Adam(param_groups,weight_decay=5e-4)

if pre_eeg_true:
    for batch_id in range(epoch):
        tem_time = time.time()

        model.train()
        opt.zero_grad()

        src_data = dataloader.get_item_nca(output_type='train', batch=batch, way=10)

        P1, P2, N1 = src_data

        P1 = torch.tensor([item.cpu().detach().numpy() for item in X]).cuda().float()
        P1 = P1.reshape(batch, 3, 224, 224)
        pred_1 = model(P1)

        P2 = torch.tensor([item.cpu().detach().numpy() for item in X]).cuda().float()
        P2 = P2.reshape(batch, 3, 224, 224)
        pred_2 = model(P2)

        N1 = torch.tensor([item.cpu().detach().numpy() for item in X]).cuda().float()
        N1 = N1.reshape(batch, 3, 224, 224)
        pred_N = model(N1)
        loss, corr_ = Loss(pred_1, pred_2, pred_N)
        loss_ = loss.item()

        train_loss += loss_
        train_correct += corr_
        loss.backward()
        opt.step()


        print('\nTest set: Avg. loss: {:.4f}, corr: {:.4f}, Epoch: {}/{} \n'.format(
            loss_, corr_, batch_id, epoch))

        if (batch_id + 1) % 500 == 0:
            writer.add_scalar('training/correct',
                              train_correct/500.0,
                              batch_id)
            writer.add_scalar('training/loss',
                              train_loss/500.0,
                              batch_id)
            train_correct = 0
            train_loss = 0
            if (batch_id + 1) % 500 == 0:
                for p in opt.param_groups:
                    p['lr'] *= 0.9
            test_correct_1 = 0
            test_correct_5 = 0
            test_loss = 0
            with torch.no_grad():
                for i in range(100):
                    src_data = dataloader.get_item_nca(output_type='test', batch=batch, way=way)
                    P1, P2, N1 = src_data

                    P1 = torch.tensor([item.cpu().detach().numpy() for item in p1]).cuda().float()
                    P2 = torch.tensor([item.cpu().detach().numpy() for item in p2]).cuda().float()
                    N1 = torch.tensor([item.cpu().detach().numpy() for item in n1]).cuda().float()

                    P1 = P1.reshape(1, 3, 224, 224)
                    P2 = P2.reshape(1, 3, 224, 224)
                    N1 = N1.reshape(way-1, 3, 224, 224)

                    pred_P1 = model(P1)
                    pred_Query = torch.cat((model(P2), model(N1)), dim=-1)

                    loss_, corr_1 = Triplet_loss(pred_P1, pred_Query)

                    test_correct_1 += corr_
                    test_loss += loss_

            print('correct_test is {:.7f}'.format(test_correct_1))
            if test_correct_1 > correct_best and batch_id > 2000:
                correct_best = test_correct_1
                model_name_triplet = 'Triplet_{}_{}_pre_train_'.format(model_name,dataset) + str(
                    batch_id + 1) + '_correct_' + str(test_correct_1) + '.pth'
                torch.save(model.state_dict(), os.path.join(snapshot_dir, model_name_triplet))

            writer.add_scalar('testing/test_loss',
                              test_loss / 100.0,
                              batch_id)
            writer.add_scalar('testing/test_top1_acc',
                              test_correct_1 / 100.0,
                              batch_id)

            torch.cuda.empty_cache()
            torch.cuda.empty_cache()

else:
    for batch_id in range(epoch):
        tem_time = time.time()

        model.train()
        opt.zero_grad()

        src_data = dataloader.get_item_nca(output_type='train', batch=batch, way=10)

        P1, P2, N1 = src_data

        P1 = torch.tensor([item.cpu().detach().numpy() for item in X]).cuda().float()
        P1 = P1.reshape(batch, 3, 224, 224)
        pred_1 = model(P1)

        P2 = torch.tensor([item.cpu().detach().numpy() for item in X]).cuda().float()
        P2 = P2.reshape(batch, 3, 224, 224)
        pred_2 = model(P2)

        N1 = torch.tensor([item.cpu().detach().numpy() for item in X]).cuda().float()
        N1 = N1.reshape(batch, 3, 224, 224)
        pred_N = model(N1)
        loss, corr_ = Loss(pred_1, pred_2, pred_N)
        loss_ = loss.item()

        train_loss += loss_
        train_correct += corr_
        loss.backward()
        opt.step()

        print('\nTest set: Avg. loss: {:.4f}, corr: {:.4f}, Epoch: {}/{} \n'.format(
            loss_, corr_, batch_id, epoch))

        if (batch_id + 1) % 500 == 0:
            writer.add_scalar('training/correct',
                              train_correct / 500.0,
                              batch_id)
            writer.add_scalar('training/loss',
                              train_loss / 500.0,
                              batch_id)
            train_correct = 0
            train_loss = 0
            if (batch_id + 1) % 500 == 0:
                for p in opt.param_groups:
                    p['lr'] *= 0.9
            test_correct_1 = 0
            test_correct_5 = 0
            test_loss = 0
            with torch.no_grad():
                for i in range(100):
                    src_data = dataloader.get_item_nca(output_type='test', batch=batch, way=way)
                    P1, P2, N1 = src_data

                    P1 = torch.tensor([item.cpu().detach().numpy() for item in p1]).cuda().float()
                    P2 = torch.tensor([item.cpu().detach().numpy() for item in p2]).cuda().float()
                    N1 = torch.tensor([item.cpu().detach().numpy() for item in n1]).cuda().float()

                    P1 = P1.reshape(1, 3, 224, 224)
                    P2 = P2.reshape(1, 3, 224, 224)
                    N1 = N1.reshape(way - 1, 3, 224, 224)

                    pred_P1 = model(P1)
                    pred_Query = torch.cat((model(P2), model(N1)), dim=-1)

                    loss_, corr_1 = Triplet_loss(pred_P1, pred_Query)

                    test_correct_1 += corr_
                    test_loss += loss_

            print('correct_test is {:.7f}'.format(test_correct_1))
            if test_correct_1 > correct_best and batch_id > 2000:
                correct_best = test_correct_1
                model_name_triplet = 'Triplet_{}_{}_pre_train_not_'.format(model_name, dataset) + str(
                    batch_id + 1) + '_correct_' + str(test_correct_1) + '.pth'
                torch.save(model.state_dict(), os.path.join(snapshot_dir, model_name_triplet))

            writer.add_scalar('testing/test_loss',
                              test_loss / 100.0,
                              batch_id)
            writer.add_scalar('testing/test_top1_acc',
                              test_correct_1 / 100.0,
                              batch_id)

            torch.cuda.empty_cache()
            torch.cuda.empty_cache()