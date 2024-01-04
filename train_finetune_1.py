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
from model.Inception_l1 import Inception3_l1
from model.ResNet50_l1 import ResNet50_l1
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn import Parameter

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

# def l2_norm_sim(input):
#     buffer = torch.pow(input, 2)
#     normp = torch.sum(buffer, 1).add_(1e-12)
#     norm = torch.sqrt(normp)
#     return norm

def binarize(T, nb_classes, smoothing_const = 0.3):
    # Optional: BNInception uses label smoothing, apply it for retraining also
    # "Rethinking the Inception Architecture for Computer Vision", p. 6
    import sklearn.preprocessing
    T = np.asarray(T)
    T = sklearn.preprocessing.label_binarize(

        T, classes = range(0, nb_classes)
    )
    T = -T * (1 - smoothing_const) * (10/nb_classes)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    T = torch.FloatTensor(T).cuda()
    return T

class ProxyNCA(torch.nn.Module):
    def __init__(self,
        nb_classes,
        sz_embedding,
        batch_size,
        smoothing_const = 0.4,
        scaling_x = 1,
        scaling_p = 3,
        pre_train = False,
    ):
        torch.nn.Module.__init__(self)
        # initialize proxies s.t. norm of each proxy ~1 through div by 8
        # i.e. proxies.norm(2, dim=1)) should be close to [1,1,...,1]
        # TODO: use norm instead of div 8, because of embedding size
        self.proxies = Parameter(torch.randn(nb_classes, sz_embedding) / 8).cuda()
        self.nb_class = nb_classes
        self.smoothing_const = smoothing_const
        self.scaling_x = scaling_x
        self.scaling_p = scaling_p
        self.batch_size = batch_size

    def add_proxy(self, proxies):
        self.proxies = proxies

    def forward(self, X, T):
        T_ = T
        P = self.proxies
        D = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        T = binarize(T=T, nb_classes=self.nb_class)

        loss = torch.sum(-T * F.log_softmax(-D, -1), -1)


        corr = 0

        _,max_dist = torch.max(D,dim=-1)
        T_ = torch.tensor(T_,dtype=torch.float32).cuda()

        Corr = torch.eq(max_dist,T_)
        for i in range(len(Corr)):
            if Corr[i]:
                corr += 1/self.batch_size
        return loss.mean(),corr

def ProxyNCA_test(positive1, positive2, negative):
    p1 = F.normalize(positive1, p=2, dim=-1)
    index = random.randint(0,9)
    n1 = torch.cat([negative[:index],positive2,negative[index:]])
    n1 = F.normalize(n1, p=2, dim=-1)

    D = F.cosine_similarity(p1,n1,dim=-1)

    corr_1 = 0
    corr_2 = 0
    # with open('train_cosine_loss.txt', "a", encoding="utf-8") as f:
    #     f.write("D:\n")
    #     f.write(str(D))
    #     f.write('\n')
    loss_weight = [0.1]*10
    loss_weight[index] = -0.9
    loss = -torch.tensor(loss_weight).cuda() * F.log_softmax(-D, -1)

    big_num = 0
    for k in range(D.shape[0]):
        if D[k] <= D[index]:
            big_num += 1
    if big_num == 10:
        corr_1 += 1
    if big_num >= 6:
        corr_2 += 1
    return loss.sum(), corr_1, corr_2

epoch = 0
epoch = 8000
correct_train = 0
train_loss = 0
train_correct = 0
batch = 64
way = 10
shot = 1
correct_best = 0
snapshot_dir = './exp/img_few_shot_weight/'

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

Path = './exp/eeg_img_weight/flu_encoder_img19500_correct_Inception75.pth'
# Path = './exp/eeg_img_weight/flu_encoder_img19500_correct_ResNet78.pth'
# Path_flu_img = './exp/use_pretrain/flu_encoder_img17000_correct_Inception74.pth'
# Path_time_img = './exp/use_pretrain/time_encoder_img19000_correct_ResNet80.pth'
# Path_flu_img = './exp/use_pretrain/flu_encoder_img15000_correct_ResNet78.pth'
pre_eeg_true = True
model_name = 'Inception'
if pre_eeg_true:
    if model_name == 'ResNet':
        model = ResNet50_l1(Path, num_classes=1000)
        model.train()
        model = model.cuda()

    elif model_name == 'Inception':
        model = Inception3_l1(Path, num_classes=1000)
        model.train()
        model = model.cuda()


else:
    if model_name == 'ResNet':
        model = ResNet50(num_classes=1000).cuda()
    elif model_name == 'Inception':
        model = Inception3(num_classes=1000).cuda()
    elif model_name == 'DenseNet':
        model = DenseNet(num_classes=1000).cuda()



dataset = 'fc100'
if dataset == 'fc100':
    dataloader = Few_dataset(dataset=dataset, img_path=r'/local/liuyuntao/datasets/StanfordOnlineProducts/FC100')
    Loss = ProxyNCA(nb_classes=60, batch_size=batch, sz_embedding=1000)
    Loss_test = ProxyNCA(nb_classes=40, batch_size=batch, sz_embedding=1000)
elif dataset == 'cub':
    dataloader = Few_dataset(dataset=dataset, img_path=r'/local/liuyuntao/egg_channel_3/data/cub')
    Loss = ProxyNCA(nb_classes=100, batch_size=batch, sz_embedding=1000)
    Loss_test = ProxyNCA(nb_classes=100, batch_size=batch, sz_embedding=1000)
elif dataset == 'fa':
    dataloader = Few_dataset(dataset=dataset, img_path=r'/local/liuyuntao/egg_channel_3/data/cars196')
    Loss = ProxyNCA(nb_classes=98, batch_size=batch, sz_embedding=1000)
    Loss_test = ProxyNCA(nb_classes=98, batch_size=batch, sz_embedding=1000)

if pre_eeg_true:
    writer = SummaryWriter('./tensorboard/fewshot/finetune1/{}_{}_nca_true_{}_finetune1/scalar'.format(dataset, model_name, str(datetime.datetime.now().date())))
else:
    writer = SummaryWriter('{}_{}_nca_false_{}/scalar'.format(dataset, model_name, str(datetime.datetime.now().date())))

if pre_eeg_true:
    param_groups = []
    param_groups.append({'params': list(set(model.parameters()).difference(set(model.fc.parameters()))), 'lr': 2e-5})
    param_groups.append({'params': model.fc.parameters(), 'lr': 1e-4})
    param_groups.append({'params': Loss.parameters(), 'lr':1e-4})

    opt = optim.Adam(param_groups,weight_decay=5e-4)

else:
    param_groups = [{'params': list(set(model.parameters()).difference(set(model.fc.parameters()))), 'lr': 2e-4},
                      {'params': model.fc.parameters(), 'lr': 1e-4}]
    param_groups.append({'params': Loss.parameters()})
    opt = optim.Adam(param_groups,weight_decay=5e-4)



def proxy_generate(proxy, pre_train, model_list):
    proxy_tensor = torch.tensor([item.cpu().detach().numpy() for item in proxy]).cuda().float()
    if pre_train:
        for i in range(len(proxy)):
            if i == 0:
                input_img = proxy_tensor[i:i+1]
                input_img = input_img.reshape(1, 3, 224, 224)
                pred_0 = torch.cat((model_list[0](input_img), model_list[0](input_img)), dim=-1)
                return_proxy = pred_0
            else:
                input_img = proxy_tensor[i:i+1]
                input_img = input_img.reshape(1, 3, 224, 224)
                pred_0 = torch.cat((model_list[0](input_img), model_list[0](input_img)), dim=-1)
                return_proxy = torch.cat((return_proxy,pred_0), dim=0)
    else:
        for i in range(len(proxy)):
            if i == 0:
                input_img = proxy_tensor[i:i+1]
                input_img = input_img.reshape(1, 3, 224, 224)
                pred_0 = model_list[0](input_img)
                return_proxy = pred_0
            else:
                input_img = proxy_tensor[i:i+1]
                input_img = input_img.reshape(1, 3, 224, 224)
                pred_0 = model_list[0](input_img)
                return_proxy = torch.cat((return_proxy,pred_0), dim=0)

    return return_proxy



if pre_eeg_true:
    for batch_id in range(epoch):
        tem_time = time.time()

        model.train()
        opt.zero_grad()

        src_data = dataloader.get_item_nca(output_type='train', batch=batch, way=10)

        X,T = src_data

        X = torch.tensor([item.cpu().detach().numpy() for item in X]).cuda().float()
        X = X.reshape(batch, 3, 224, 224)
        pred_X = model(X)

        loss, corr_ = Loss(pred_X,T)
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
            if (batch_id + 1) % 500 == 0 and train_correct / 500.0 > 0.8:
                for p in opt.param_groups:
                    p['lr'] *= 0.9
            test_correct_1 = 0
            test_correct_5 = 0
            test_loss = 0
            with torch.no_grad():
                for i in range(100):
                    src_data = dataloader.get_item_nca(output_type='test', batch=batch, way=10)
                    p1, p2, n1 = src_data

                    P1 = torch.tensor([item.cpu().detach().numpy() for item in p1]).cuda().float()
                    P2 = torch.tensor([item.cpu().detach().numpy() for item in p2]).cuda().float()
                    N1 = torch.tensor([item.cpu().detach().numpy() for item in n1]).cuda().float()

                    P1 = P1.reshape(1, 3, 224, 224)
                    P2 = P2.reshape(1, 3, 224, 224)
                    N1 = N1.reshape(way-1, 3, 224, 224)

                    pred_p1 = model(P1)
                    pred_p2 = model(P2)
                    pred_n1 = model(N1)

                    loss_, corr_1, corr_2 = ProxyNCA_test(pred_p1, pred_p2, pred_n1)

                    test_correct_1 += corr_
                    test_correct_5 += corr_2
                    test_loss += loss_

            print('correct_test is {:.7f}'.format(test_correct_1))
            if test_correct_1 > correct_best and batch_id > 2000:
                correct_best = test_correct_1
                model_name_ = 'ProxyNCA_few_shot_cosine_t_{}_{}_batch_pre_train_l1'.format(model_name,dataset) + str(
                    batch_id + 1) + '_correct_' + str(test_correct_1) + '.pth'
                torch.save(model.state_dict(), os.path.join(snapshot_dir, model_name_))

            writer.add_scalar('testing/test_loss',
                              test_loss / 100.0,
                              batch_id)
            writer.add_scalar('testing/test_top1_acc',
                              test_correct_1 / 100.0,
                              batch_id)
            writer.add_scalar('testing/test_top5_acc',
                              test_correct_5 / 100.0,
                              batch_id)

            torch.cuda.empty_cache()
            torch.cuda.empty_cache()


else:
    for batch_id in range(epoch):
        tem_time = time.time()

        model.train()
        opt.zero_grad()

        src_data = dataloader.get_item_nca(output_type='train', batch=batch, way=10)

        X, T = src_data

        X = torch.tensor([item.cpu().detach().numpy() for item in X]).cuda().float()
        X = X.reshape(batch, 3, 224, 224)
        pred_X = model(X)

        loss, corr_ = Loss(pred_X, T)
        loss_ = loss.item()
        train_loss += loss_
        train_correct += corr_

        loss.backward()
        opt.step()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

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
            if (batch_id + 1) % 500 == 0 and train_correct / 500.0 > 0.8:
                for p in opt.param_groups:
                    p['lr'] *= 0.9
            test_correct_1 = 0
            test_correct_5 = 0
            test_loss = 0
            way = 10
            with torch.no_grad():
                for i in range(100):
                    src_data = dataloader.get_item_nca(output_type='test', batch=batch, way=way)
                    p1,p2,n1 = src_data

                    P1 = torch.tensor([item.cpu().detach().numpy() for item in p1]).cuda().float()
                    P2 = torch.tensor([item.cpu().detach().numpy() for item in p2]).cuda().float()
                    N1 = torch.tensor([item.cpu().detach().numpy() for item in n1]).cuda().float()

                    P1 = P1.reshape(1, 3, 224, 224)
                    P2 = P2.reshape(1, 3, 224, 224)
                    N1 = N1.reshape(way-1, 3, 224, 224)

                    pred_p1 = model(P1)
                    pred_p2 = model(P2)
                    pred_n1 = model(N1)

                    loss_, corr_1, corr_2 = ProxyNCA_test(pred_p1, pred_p2, pred_n1)
                    test_correct_1 += corr_1
                    test_correct_5 += corr_2
                    test_loss += loss_

            print('correct_test is {:.7f}'.format(test_correct_1))
            if test_correct_1  > correct_best and batch_id > 2000:
                correct_best = test_correct_1
                model_name_ = 'ProxyNCA_few_shot_cosine_{}_{}_batch_pre_train_not'.format(model_name,dataset) + str(
                    batch_id + 1) + '_correct_' + str(test_correct_1) + '.pth'
                torch.save(model.state_dict(), os.path.join(snapshot_dir, model_name_))

            writer.add_scalar('testing/test_loss',
                              test_loss /100.0,
                              batch_id)
            writer.add_scalar('testing/test_top1_acc',
                              test_correct_1 /100.0,
                              batch_id)
            writer.add_scalar('testing/test_top5_acc',
                              test_correct_5 / 100.0,
                              batch_id)

            torch.cuda.empty_cache()
