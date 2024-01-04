import copy
import time

import numpy as np
import torch
import random
from torch.utils import data
import csv
import os
from dataset.filter import get_fft_values,butter_bandpass_filter,new_filter
import cv2
import sys

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

    def sss_exit(self):
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


class EEG_IMG_dataset():
    def __init__(self, eeg_path = r'/local/liuyuntao/datasets/Mind3/MindBigData-Imagenet',fluent_path =r'/local/liuyuntao/datasets/Mind3/MindBigData-Imagenet-v1.0-Imgs', img_path = r'/local/ImageNet_Raw/train/'):
        self.file = []
        self.path_e = eeg_path
        self.path_i = img_path
        self.crop_size = (224, 224)
        self.path_s = fluent_path
        self.kind_list = []
        self.kind_belong = []

        with open('dataset/data_load_2.txt','r') as f:
            h1 = f.readline()
            while h1 != '':
                h1 = h1.split(',')
                self.kind_list.append(h1[0])
                self.kind_belong.append(h1[-1][:-1])
                h1 = f.readline()

        all_csv_file = os.listdir(self.path_e)
        i = 0
        index_class = 0
        self.dict_class = {}
        for file in all_csv_file:
            i += 1
            egg_img_dict = {}

            #获取脑电波时序数据
            egg_address = self.path_e + '/' + file
            egg_img_dict['egg_file'] = egg_address

            #获取脑电波频域数据
            fluent_file = self.path_s + '/' + file[:-3] + 'png'
            egg_img_dict['fluent_file'] = fluent_file

            #获取该样本所属小类
            file = file.split('_')
            image_name = file[3] + '_' + file[4]
            egg_img_dict['class'] = file[3]

            #获取该样本所属大类
            index_kind = self.kind_list.index(file[3])
            egg_img_dict['kind'] = self.kind_belong[index_kind]

            if egg_img_dict['class'] in self.dict_class.keys():
                egg_img_dict['class_index'] = self.dict_class[file[3]]
            else:
                self.dict_class[file[3]] = index_class
                egg_img_dict['class_index'] = index_class
                index_class += 1

            img_file_path_raw = [file[3], image_name]

            img_address = self.path_i + img_file_path_raw[0] + '/' + img_file_path_raw[
                1] + '.JPEG'
            egg_img_dict['img_file'] = img_address

            self.file.append(egg_img_dict)

    def return_file(self):
        return self.file

    def len(self):
        return len(self.file)

    def read_csv(self,address):
        channel_EGG_data = []
        with open(address, mode='r') as f:
            reader = csv.reader(f)
            for i, rows in enumerate(reader):
                length = len(rows) - 320
                rows = rows[length:]
                channel_EGG_data.append(rows)
        for row in range(len(channel_EGG_data)):
            for line in range(len(channel_EGG_data[0])):
                channel_EGG_data[row][line] = float(channel_EGG_data[row][line])
        return channel_EGG_data

    def gauss_noisy(self, x):
        mu = 0
        sigma = 0.05
        for i in range(len(x)):
            x[i] += random.gauss(mu, sigma)


    def filter_(self,egg_data):
        L2 = []
        sys.stdout = open(os.devnull, 'w')
        flu_1 = new_filter(egg_data.tolist(),4.0,0.5)
        L2.append(flu_1.tolist())
        flu_2 = new_filter(egg_data.tolist(),8.0,4.0)
        L2.append(flu_2.tolist())
        flu_3 = new_filter(egg_data.tolist(),13.0,8.0)
        L2.append(flu_3.tolist())
        flu_4 = new_filter(egg_data.tolist(),32.0,13.0)
        L2.append(flu_4.tolist())
        flu_5 = new_filter(egg_data.tolist(),63.5,32.0)
        L2.append(flu_5.tolist())
        sys.stdout = sys.__stdout__

        return np.asarray(L2, np.float32)

    def getitem(self, get_type, negative_len, batch_size):
        #脑电波的预处理
        data_EEG_Pos = []
        data_EEG_Neg = []
        data_IMG = []
        for i in range(batch_size):
            if get_type == 'train':
                index = random.randint(0, 8000)
                while self.file[index]['class_index'] >= 400:
                    index = random.randint(0, 8000)

            if get_type == 'test':
                index = random.randint(0, 8000)
                while self.file[index]['class_index'] <= 400:
                    index = random.randint(0, 8000)

            data_once_EGG = self.read_csv(self.file[index]['egg_file'])
            med_row = np.median(data_once_EGG, axis=1)
            data_once_EGG = data_once_EGG - np.matrix(med_row).T

            #原始时序数据e0
            e0 = np.asarray(data_once_EGG,np.float32)
            noisy = np.random.normal(0, 1, (5, 320))
            e1 = e0 + noisy * 0.01

            #时序滤波数据t0
            t0 = self.filter_(e1)
            data_EEG_Pos.append(t0)

            data_EEG_Neg_once = []
            for j in range(negative_len):
                index_ = random.randint(0, 8000)
                if get_type == 'train':
                    while self.file[index]['class'] == self.file[index_]['class'] or self.file[index]['class_index'] >= 400 or self.file[index]['kind'] == self.file[index_]['kind']:
                        index_ = random.randint(0, 8000)
                if get_type == 'test':
                    while self.file[index]['class'] == self.file[index_]['class'] or self.file[index]['class_index'] < 400 or self.file[index]['kind'] == self.file[index_]['kind']:
                        index_ = random.randint(0, 8000)

                #
                # for file_negative in self.file:
                #     if file_negative['class'] == self.file[index_]['class']:
                #         negative_index.append(self.file.index(file_negative))
                #
                # id = 0
                # for index_ in negative_index:
                #     id += 1
                #     if id == 6:
                #         break
                data_once_EGG_ = self.read_csv(self.file[index_]['egg_file'])

                e_n = np.asarray(data_once_EGG_, np.float32)
                med_row = np.median(e_n, axis=1)
                e_n = e_n - np.matrix(med_row).T
                noisy = np.random.normal(0, 1, (5, 320))
                e_n = e_n + noisy * 0.01

                sn_0 = self.filter_(e_n)

                data_EEG_Neg_once.append(sn_0)

            data_EEG_Neg.append(data_EEG_Neg_once)


            #图片的预处理
            img_data = cv2.imread(self.file[index]['img_file'])
            img_data = cv2.resize(img_data, self.crop_size, interpolation=cv2.INTER_CUBIC)
            img_data = np.asarray(img_data, np.float32)

            data_IMG.append(img_data)


        return data_EEG_Pos, data_EEG_Neg, data_IMG

