import copy
import time

import numpy as np
import torch
import random
import os
import cv2
import sys

class Few_dataset():
    def __init__(self, dataset, img_path ):
        self.path = img_path
        self.dataset = dataset
        if self.dataset == 'cub':
            self.cub()
        elif self.dataset == 'cars196':
            self.cars196()
        elif self.dataset == 'fc100':
            self.fc100()
        self.crop_size = (224, 224)

    def fc100(self):
        self.file = []
        for _ in range(100):
            self.file.append([])
        path = self.path + "/train"
        classes = os.listdir(path)
        i = 0
        for oneClass in classes:
            if oneClass == ".DS_Store":
                continue
            classPath = os.listdir(path + "/" + oneClass)
            for oneClassPath in classPath:
                self.file[i].append(path + "/" + oneClass + "/" + oneClassPath)
            i += 1

        path = self.path + "/val"
        classes = os.listdir(path)
        for oneClass in classes:
            if oneClass == ".DS_Store":
                continue
            classPath = os.listdir(path + "/" + oneClass)
            for oneClassPath in classPath:
                self.file[i].append(path + "/" + oneClass + "/" + oneClassPath)
            i += 1

        path = self.path + "/test"
        classes = os.listdir(path)
        for oneClass in classes:
            if oneClass == ".DS_Store":
                continue
            classPath = os.listdir(path + "/" + oneClass)
            for oneClassPath in classPath:
                self.file[i].append(path + "/" + oneClass + "/" + oneClassPath)
            i += 1

    def cub(self):
        self.file = []
        for _ in range(200):
            self.file.append([])
        with open('{}/CUB_200_2011/images.txt'.format(self.path), 'r') as f:
            h1 = f.readline()
            while h1 != '':
                h = h1.split(' ')
                h2 = h[1].split('.')
                self.file[int(h2[0])-1].append('{}/CUB_200_2011/images/{}'.format(self.path,h[1][:-1]))
                h1 = f.readline()

    def cars196(self):
        self.file = []
        for _ in range(196):
            self.file.append([])
        with open('{}/devkit/train_perfect_preds.txt'.format(self.path), 'r') as f:
            h1 = f.readline()
            id = 1
            while h1 != '':
                self.file[int(h1[:-1])-1].append('{}/cars_train/{}.jpg'.format(self.path,str(id).zfill(5)))
                id += 1
                h1 = f.readline()

    def ImageNet(self):
        self.file = []
        for _ in range(1000):
            self.file.append([])
        train_img_path = os.listdir(self.path)
        id = 0
        for document in train_img_path:
            train_img_one_path = self.path + '/' + document
            once_img_path_all = os.listdir(train_img_one_path)
            for name in once_img_path_all:
                once_img_path = train_img_one_path + '/' + name
                self.file[id].append(once_img_path)
            id += 1


    def random_read_picture(self,path):
        img = cv2.imread(path)
        img = cv2.resize(img, self.crop_size, interpolation=cv2.INTER_CUBIC)
        img = np.asarray(img, np.float32)

        num_noise = 100
        for num in range(num_noise):
            x = random.randint(0, img.shape[0] - 1)
            y = random.randint(0, img.shape[1] - 1)
            if num % 2 == 0:
                img[x, y] = 0
            else:
                img[x, y] = 255

        # rand_num = random.randint(0,6)
        # if rand_num == 1:
        #     angle = 90
        #     height, width = img.shape[:2]
        #     if height > width:
        #         center = (height / 2, height / 2)
        #     else:
        #         center = (width / 2, width / 2)
        #     mata = cv2.getRotationMatrix2D(center, angle, scale=1)
        #     img = cv2.warpAffine(img, mata, (height, width), borderValue=(0, 0, 0))
        # elif rand_num == 2:
        #     angle = 180
        #     height, width = img.shape[:2]
        #     if height > width:
        #         center = (height / 2, height / 2)
        #     else:
        #         center = (width / 2, width / 2)
        #     mata = cv2.getRotationMatrix2D(center, angle, scale=1)
        #     img = cv2.warpAffine(img, mata, (height, width), borderValue=(0, 0, 0))
        # elif rand_num == 3:
        #     angle = 270
        #     height, width = img.shape[:2]
        #     if height > width:
        #         center = (height / 2, height / 2)
        #     else:
        #         center = (width / 2, width / 2)
        #     mata = cv2.getRotationMatrix2D(center, angle, scale=1)
        #     img = cv2.warpAffine(img, mata, (height, width), borderValue=(0, 0, 0))
        # elif rand_num == 4:
        #     img = cv2.flip(img, 1)
        # elif rand_num == 5:
        #     img = cv2.flip(img, 0)
        # elif rand_num == 6:
        #     img = cv2.GaussianBlur(img, (3, 3), sigmaX=1)

        img = torch.tensor(img,dtype=torch.float32)

        return img

    def get_item(self, output_type, way, shot, batch):
        if self.dataset == 'cub':
            train_length = [0,99]
            test_length = [100,199]
        elif self.dataset == 'cars196':
            train_length = [0, 97]
            test_length = [98, 195]
        elif self.dataset == 'fc100':
            train_length = [0, 59]
            test_length = [60, 99]

        if output_type == 'train':
            batch_positive_1 = []
            batch_positive_2 = []
            batch_negative = []
            for i in range(batch):
                read_img_id = "train:"
                positive = []
                negative = []
                index_ = random.randint(train_length[0], train_length[1])
                choose_pic = [random.randint(0, 10) for _ in range(2)]
                positive_1 = self.random_read_picture(self.file[index_][choose_pic[0]])
                positive_2 = self.random_read_picture(self.file[index_][choose_pic[1]])

                way_class = [index_]

                for i in range(way):
                    negative_once = []
                    index_n = random.randint(train_length[0], train_length[1])
                    while index_n in way_class:
                        index_n = random.randint(train_length[0], train_length[1])
                    way_class.append(index_n)
                    choose_pic = [random.randint(0, 10) for _ in range(shot)]
                    # read_img_id += (str(index_n) +":" + str(choose_pic[0]) + " ")
                    for j in range(shot):
                        negative_once.append(self.random_read_picture(self.file[index_n][choose_pic[j]]).tolist())
                    negative.append(negative_once)

                # with open('./dataset_check.txt', "a", encoding="utf-8") as f:
                #     f.write(read_img_id + "\n")
                batch_positive_1.append(positive_1)
                batch_positive_2.append(positive_2)
                batch_negative.append(negative)
            # batch_positive_1 = torch.tensor(batch_positive_1)
            # batch_positive_2 = torch.tensor(batch_positive_1)
            return batch_positive_1,batch_positive_2, batch_negative

        if output_type == 'test':
            negative = []
            index_ = random.randint(test_length[0], test_length[1])

            positive_img_list_len = len(self.file[index_])
            choose_pic = [random.randint(0, positive_img_list_len - 1) for _ in range(2)]

            # read_img_id = "test:"
            # read_img_id += (str(index_) + ":" + str(choose_pic[0]) + " ")
            # read_img_id += (str(index_) + ":" + str(choose_pic[1]) + " ")

            positive_1 = self.random_read_picture(self.file[index_][choose_pic[0]])
            positive_2 = self.random_read_picture(self.file[index_][choose_pic[1]])

            way_class = [index_]

            for i in range(way):
                negative_once = []
                index_n = random.randint(test_length[0], test_length[1])
                while index_n in way_class:
                    index_n = random.randint(test_length[0], test_length[1])
                way_class.append(index_n)
                positive_img_list_len = len(self.file[index_n])
                choose_pic = [random.randint(0, positive_img_list_len - 1) for _ in range(shot)]
                # read_img_id += (str(index_n) + ":" + str(choose_pic[0]) + " ")
                for j in range(shot):
                    negative_once.append(self.random_read_picture(self.file[index_n][choose_pic[j]]).tolist())
                negative.append(negative_once)
            # with open('./dataset_check.txt', "a", encoding="utf-8") as f:
            #     f.write(read_img_id + "\n")


            return positive_1, positive_2, negative

    def get_item_nca(self, output_type, batch, way):
        if self.dataset == 'cub':
            train_length = [0,99]
            test_length = [100,199]
        elif self.dataset == 'cars196':
            train_length = [0, 97]
            test_length = [98, 195]
        elif self.dataset == 'ImageNet':
            train_length = [0, 399]
            test_length = [400, 799]
        elif self.dataset == 'fc100':
            train_length = [0,59]
            test_length = [59,99]

        if output_type == 'train':
            T = []
            X = []
            for i in range(batch):
                index_ = random.randint(train_length[0], train_length[1])
                choose_pic = [random.randint(0, 10) for _ in range(1)]
                X.append(self.random_read_picture(self.file[index_][choose_pic[0]]))
                T.append(index_)
            return X,T

        if output_type == 'test':
            p1 = []
            p2 = []
            n1 = []
            T = []
            index_ = random.randint(test_length[0], test_length[1])
            T.append(index_)
            choose_pic = random.sample(range(0,15),2)
            p1.append(self.random_read_picture(self.file[index_][choose_pic[0]]))
            p2.append(self.random_read_picture(self.file[index_][choose_pic[1]]))

            for i in range(way-1):
                index_ = random.randint(test_length[0], test_length[1])
                while index_ in T:
                    index_ = random.randint(test_length[0], test_length[1])
                T.append(index_)
                choose_pic = random.sample(range(0,10),1)
                n1.append(self.random_read_picture(self.file[index_][choose_pic[0]]))
            # T = []
            # X = []
            # for i in range(batch):
            #     index_ = random.randint(test_length[0], test_length[1])
            #     choose_pic = [random.randint(1, 11) for _ in range(1)]
            #     X.append(self.random_read_picture(self.file[index_][choose_pic[0]]))
            #     T.append(index_-test_length[0])
            #
            # return X, T

            return p1,p2,n1

    def get_proxy(self, output_type):
        if self.dataset == 'cub':
            train_length = [0, 100]
            test_length = [100, 200]
        elif self.dataset == 'cars196':
            train_length = [0, 98]
            test_length = [98, 196]
        elif self.dataset == 'ImageNet':
            train_length = [0, 400]
            test_length = [400, 800]

        if output_type == 'train':
            X = []
            for i in range(train_length[0],train_length[1]):
                X.append(self.random_read_picture(self.file[i][0]))

        if output_type == 'test':
            X = []
            for i in range(test_length[0], test_length[1]):
                X.append(self.random_read_picture(self.file[i][0]))

        return X

    def get_item_all(self, class_id):
        X = []
        for i in range(len(self.file[class_id])):
            X.append(self.random_read_picture(self.file[class_id][i]))

        return X

    def return_file(self):
        print(self.file)

    def get_item_triplet(self, output_type, way, shot, batch):
        if self.dataset == 'cub':
            train_length = [0, 99]
            test_length = [100, 199]
        elif self.dataset == 'cars196':
            train_length = [0, 97]
            test_length = [98, 195]
        elif self.dataset == 'ImageNet':
            train_length = [0, 399]
            test_length = [400, 799]

        if output_type == 'train':
            P1 = []
            P2 = []
            N1 = []
            for i in range(batch):
                index = random.randint(train_length[0], train_length[1])
                choose_pic = [random.randint(0, 10) for _ in range(2)]

                index_ = random.randint(test_length[0], test_length[1])
                while index_ != index:
                    index_ = random.randint(test_length[0], test_length[1])
                P1.append(self.random_read_picture(self.file[index][choose_pic[0]]))
                P2.append(self.random_read_picture(self.file[index][choose_pic[1]]))
                N1.append(self.random_read_picture(self.file[index_][choose_pic[1]]))

            return P1, P2, N1

        if output_type == 'test':
            P1 = []
            P2 = []
            N1 = []
            for i in range(batch):
                index = random.randint(train_length[0], train_length[1])
                choose_pic = [random.randint(0, 10) for _ in range(2)]


                P1.append(self.random_read_picture(self.file[index][choose_pic[0]]))
                P2.append(self.random_read_picture(self.file[index][choose_pic[1]]))
                for j in range(way-1):
                    index_ = random.randint(test_length[0], test_length[1])
                    while index_ != index:
                        index_ = random.randint(test_length[0], test_length[1])
                    N1.append(self.random_read_picture(self.file[index_][choose_pic[0]]))

            return P1, P2, N1

    def get_item_index(self, classIndex, index):
        if self.dataset == 'cub':
            train_length = [0, 100]
            test_length = [100, 200]
        elif self.dataset == 'cars196':
            train_length = [0, 98]
            test_length = [98, 196]

        X = []
        X.append(self.random_read_picture(self.file[classIndex][index]))
        return X
