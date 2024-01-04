from tensorboardX import SummaryWriter
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset.EEG_IMG_dataset import EEG_IMG_dataset
from model.fluent_model import EGG_encoder_s
from model.Inception import Inception3
from model.ResNet50 import ResNet50
import torch.nn.functional as F
import matplotlib.pyplot as plt

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class Fluent_loss(nn.Module):
    def __init__(self):
        super(Fluent_loss, self).__init__()
        self.temperature = 0.01

    def forward(self, feature_s_1, feature_s_2, feature_f):
        correct_loss = 0
        top = F.linear(l2_norm(feature_s_1), l2_norm(feature_s_2))
        top = torch.div(top, self.temperature)
        top = torch.exp(top)
        bot = torch.tensor(0, dtype=torch.float32)
        for feature_f_n in feature_f:
            bot_ = F.linear(l2_norm(feature_f_n), l2_norm(feature_s_2))
            bot_ = torch.div(bot_, self.temperature)
            bot_ = torch.exp(bot_)
            bot = torch.add(bot_, bot)

        bot = bot + top
        bot = torch.div(bot, len(feature_f) + 1)

        if top.item() > bot.item():
            correct_loss += 1

        logit_get = torch.div(top, bot)

        loss = (- torch.log(logit_get) + torch.tensor(1.0))
        loss.requires_grad_()

        return loss, correct_loss


def Loss_test(feature_t_1, feature_t_2, feature_f):
    top = F.linear(l2_norm(feature_t_1), l2_norm(feature_t_2))
    top = torch.exp(top)
    bot = F.linear(l2_norm(feature_t_1), l2_norm(feature_f))
    bot = torch.exp(bot)

    logit_get = torch.div(top, bot)
    loss = - torch.log(logit_get)

    return loss


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

writer = SummaryWriter('tensorboard/eeg_img/ResNet_pretrain/scalar')

model = EGG_encoder_s(num_classes=1000)

model.train()
model = model.cuda()

model_name = "ResNet"
if model_name == "ResNet":
    model_img = ResNet50(num_classes=1000)
    model_img.train()
    model_img = model_img.cuda()
elif model_name == "Inception":
    model_img = Inception3(num_classes=1000)
    model_img.train()
    model_img = model_img.cuda()


dataloader = EEG_IMG_dataset()

optimizer = optim.Adam([{'params': model.parameters()}, {'params': model_img.parameters()}],
                         lr=1e-4)

loss_flu = Fluent_loss()

epoch = 0
all_epoch = 8000
batch_size = 1
losses_f_train = []
losses_t_train = []
losses_f_test = []
losses_t_test = []
correct_f_train = []
correct_t_train = []
correct_f_test = []
correct_t_test = []
correct_best = 0
snapshot_dir = './exp/eeg_img_weight/'
correct_train = 0

negative_len = 5
loss_all = 0
ever_f_correct_best = 0
ever_t_correct_best = 0
for batch_id in range(all_epoch):
    tem_time = time.time()

    model_img.train()
    model.train()
    optimizer.zero_grad()

    src_data = dataloader.getitem(get_type='train', negative_len=negative_len, batch_size=batch_size)

    positive, negative, image = src_data

    image = torch.tensor(image).cuda().float()
    image = image.reshape((batch_size, 3, 224, 224))
    img_feature = model_img(image)

    et = torch.tensor(positive).cuda().float()
    pred_pos = model(et)

    pred_neg = []
    ena = torch.tensor(negative).cuda().float()
    for i in range(negative_len):
        en = ena[:,i,:,:,:]
        pred_neg.append(model(en))

    Loss, correct = loss_flu(pred_pos, img_feature, pred_neg)
    loss = Loss.item()
    writer.add_scalar('training/loss',
                      loss,
                      batch_id)
    Loss.backward()
    optimizer.step()

    correct_train += correct
    print('\nTrain set: loss: {:.4f}, correct: {:.4f}, Epoch: {}/{} \n'.format(
        Loss.item(), correct, batch_id, all_epoch))

    loss_all += loss
    if (batch_id + 1) % 500 == 0:
        print('\nTest set: Avg. loss: {:.4f}, correct: {:.4f}, Epoch: {}/{} \n'.format(
            loss/500.0, correct_train/500.0, batch_id, all_epoch))
        writer.add_scalar('training/correct',
                          correct_train / 500.0,
                          batch_id / 500)
        writer.add_scalar('training/loss',
                          loss_all / 500.0,
                          batch_id / 500)
        if correct_train >= 70.0 :
            for p in optimizer.param_groups:
                p['lr'] *= 0.9
        correct_train = 0
        loss_all = 0

    if (batch_id + 1) % 500 == 0:
        correct_test = 0
        loss_test = 0


        for i in range(100):
            src_data = dataloader.getitem('test', 1, 1)
            positive, negative, image = src_data

            et = torch.tensor(positive).cuda().float()
            et = et.reshape((1, 5, 5, 320))
            pred_pos = model(et)

            image = torch.tensor(image).cuda().float()
            image = image.reshape((1, 3, 224, 224))
            img_feature = model_img(image)

            ef = torch.tensor(negative[0]).cuda().float()
            ef = ef.reshape(1, 5, 5, 320)
            pred_neg = model(ef)

            loss_test_once = Loss_test(pred_pos, img_feature, pred_neg)
            loss_test += loss_test_once.item()

            if F.linear(l2_norm(pred_pos), l2_norm(img_feature)) > F.linear(l2_norm(pred_neg), l2_norm(img_feature)):
                correct_test += 1

        print('test correct is {:.7f}'.format(correct_test / 100.0))

        if correct_test / 100.0 > correct_best:
            correct_best = correct_test / 100.0
            weight_name = 'IMG' + repr(batch_id + 1) + '_correct_{}'.format(model_name) + repr(int(correct_test)) + '.pth'
            torch.save(model_img.state_dict(), os.path.join(snapshot_dir, weight_name))
            weight_name = 'EEG' + repr(batch_id + 1) + '_correct_{}'.format(model_name) + repr(int(correct_test)) + '.pth'
            torch.save(model.state_dict(), os.path.join(snapshot_dir, weight_name))

        writer.add_scalar('testing/loss',
                          loss_test / 100.0,
                          batch_id / 500)
        writer.add_scalar('testing/correct',
                          correct_test / 100.0,
                          batch_id / 500)

