import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import utils as vutils
import random
import os
from torch.utils.data import Dataset

class Pad2Square(object):
    def __init__(self):
        pass
    def __call__(self, img):
        return transforms.CenterCrop(max(img.size))(img)


def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)  # C x H x W  ---> H x W x C

    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy() * 255

    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()

    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))

    return img


def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    vutils.save_image(input_tensor, filename)


class ImageDataset(Dataset):

    def __init__(self, train, validation, test):


        self.train = train
        self.validation = validation
        self.test = test

        self.imgs_path = list(sorted(os.listdir('./size_512/all')))


        self.left_image_names = self.imgs_path

        self.label0 = '0'
        # self.right_image_names = self.csv[:]['Right-Fundus']
        self.images = []

        self.train_len = int(len(self.left_image_names))


        # set the training data images and labels
        if self.train == True:
            print(f"Number of training images: {self.train_len}")


            self.Limage_names = list(self.left_image_names[:self.train_len])

            # define the training transform
            self.transform = transforms.Compose([
                # transforms.ToPILImage(),         # 这个一定要有 否则报错
                # transforms.Resize((400, 400)),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomRotation(degrees=45),
                # transforms.ToTensor(),
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),# 迁移学习的话需要减去 来去中心化（resnet 视频
                #  Github上jupter某人方案
                # transforms.ToPILImage(),
                # transforms.Resize((300, 300)),
                # transforms.RandomHorizontalFlip(p=1), transforms.RandomVerticalFlip(p=1),
                # transforms.RandomRotation(degrees=30),
                #
                # transforms.ToTensor(),
                # # transforms.RandomErasing(p=0.2),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  #resnet数据集

                # transforms.Normalize([0.5820, 0.4512, 0.4023], [0.2217, 0.1858, 0.1705]),

                #官方
                transforms.ToPILImage(),    #open 不用这个
                # Pad2Square(),
                transforms.Resize((512, 512)),
                # transforms.CenterCrop(448),  # 随机长宽比裁剪为224*224
                # transforms.Resize((256, 256)),
                # transforms.RandomResizedCrop(224),  # 随机长宽比裁剪为224*224
                # transforms.CenterCrop(224),
                # transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3),
                # transforms.RandomRotation([-180, 180]),
                # transforms.RandomAffine([-180, 180], translate=[0.1, 0.1], scale=[0.8, 1.2]),
                # # transforms.ColorJitter(brightness=0.2),   # contrast=0.5, saturation=0.5, hue=0.5
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                #
                transforms.ToTensor(),  # 转化为张量并归一化为[0-1]
                # transforms.Normalize([0.339, 0.214, 0.115], [0.352, 0.239, 0.154])
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # transforms.Normalize([0.5820, 0.4512, 0.4023], [0.2217, 0.1858, 0.1705])


            ])


    def __len__(self):
        return len(self.Limage_names)
    # def __len__(self, index):
    #     return len(self.Limage_names,self.Rimage_names)

    def __getitem__(self, index):
    # def __getitem__(self, index, index1):

        name=self.Limage_names[index]


        Limage = cv2.imread(f"./size_512/all/{self.Limage_names[index]}")  # ./DRIVE/size_512/2
        #

        Limage = cv2.cvtColor(Limage, cv2.COLOR_BGR2RGB)

        Limage = self.transform(Limage)

        name =name.replace('d', 'gd')
        Rimage = cv2.imread(f"./size_512/gall/{name}")
        # convert it from BGR to RGB
        Rimage = cv2.cvtColor(Rimage, cv2.COLOR_BGR2RGB)
        Rimage = self.transform(Rimage)
        # targets = name.split(' ')
        # key = int(name[2])  #[1, 0, 0, 0]  [0, 1, 0, 0] [0, 0, 1, 0]  [0, 0, 0, 1]
        # a = {0:[1, 0, 0, 0], 1:[0, 1, 0, 0], 2:[0, 0, 1, 0], 3:[0, 0, 1, 0]}
        zidian=[[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]
        targets =zidian[int(name[2])]



        # targets = name.split('gd').split(' ')[0]


        # print(self.Limage_names[index],name, targets)
        return {
            # 'limage': torch.tensor(Limage),
            # 'rimage': torch.tensor(Rimage),
            # 'label': torch.tensor(targets)
            'limage': torch.tensor(Limage, dtype=torch.float32),
            'rimage': torch.tensor(Rimage, dtype=torch.float32),
            'label': torch.tensor(targets, dtype=torch.float32)
        }

class ImageDatasettest(Dataset):

    def __init__(self, train, validation, test):


        self.train = train
        self.validation = validation
        self.test = test


        self.imgs_path = list(sorted(os.listdir('./size_512/all-test')))


        self.left_image_names = self.imgs_path

        self.label0 = '0'
        # self.right_image_names = self.csv[:]['Right-Fundus']
        self.images = []

        self.train_len = int(len(self.left_image_names))

        # set the training data images and labels
        if self.train == True:
            print(f"Number of training images: {self.train_len}")


            self.Limage_names = list(self.left_image_names[:self.train_len])



            # define the training transform
            self.transform = transforms.Compose([

                #官方
                transforms.ToPILImage(),    #open 不用这个
                # Pad2Square(),
                transforms.Resize((512, 512)),
                # transforms.CenterCrop(448),  # 随机长宽比裁剪为224*224
                #
                transforms.ToTensor(),  # 转化为张量并归一化为[0-1]
                # transforms.Normalize([0.339, 0.214, 0.115], [0.352, 0.239, 0.154])
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # transforms.Normalize([0.5820, 0.4512, 0.4023], [0.2217, 0.1858, 0.1705])


            ])

        elif self.validation == True:
            print(f"Number of Validation images: {self.train_len}")


            self.Limage_names = list(self.left_image_names[-self.valid_len:])
            # self.Rimage_names = list(self.right_image_names[-self.valid_len:])
            self.Rimage_names = list(self.right_image_names[-self.valid_len:])
            self.images = self.Limage_names + self.Rimage_names
            self.labels = list(self.all_labels[-self.valid_len:])


            # define the validatio transform
            self.transform = transforms.Compose([

                #官方
                transforms.ToPILImage(),    #open 不用这个
                # Pad2Square(),
                transforms.Resize((512, 512)),
                transforms.CenterCrop(448),
                # transforms.Resize((256, 256)),
                # transforms.CenterCrop(224),
                # transforms.RandomHorizontalFlip(p=1), transforms.RandomVerticalFlip(p=1),
                transforms.ToTensor(),  # 转化为张量并归一化为[0-1]
                # transforms.Normalize([0.339, 0.214, 0.115], [0.352, 0.239, 0.154])
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        elif self.test == True:
            print(f"Number of Validation images: {self.train_len}")
            # self.Limage_names0 = list(self.left_image_names0[:self.test_len0])
            # self.Limage_names1 = list(self.left_image_names1[:self.test_len1])
            # self.Limage_names2 = list(self.left_image_names2[:self.test_len2])
            # self.Limage_names3 = list(self.left_image_names3[:self.test_len3])

            # define the test transform
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),

            ])

    def __len__(self):
        return len(self.Limage_names)
    # def __len__(self, index):
    #     return len(self.Limage_names,self.Rimage_names)

    def __getitem__(self, index):
    # def __getitem__(self, index, index1):

        name=self.Limage_names[index]

        # Limage0 = cv2.imread(f"./size_512/0/{self.Limage_names[index]}")  #./DRIVE/size_512/2
        # Limage1 = cv2.imread(f"./size_512/1/{self.Limage_names[index]}")  # ./DRIVE/size_512/2
        # Limage2 = cv2.imread(f"./size_512/2/{self.Limage_names[index]}")  # ./DRIVE/size_512/2
        # Limage3 = cv2.imread(f"./size_512/3/{self.Limage_names[index]}")  # ./DRIVE/size_512/2

        Limage = cv2.imread(f"./size_512/all-test/{self.Limage_names[index]}")  # ./DRIVE/size_512/2
        #
        # Limage =cv2.resize(cv2.imread(f"../Test5_resnet/OIA-ODIR/Training Set/Images/{self.Limage_names[index]}", cv2.IMREAD_COLOR), (224, 224)).astype(np.float32)
        # # # print(Limage)
        # mean_bgr = [26.0917, 48.3404, 76.3456]
        # Limage-=np.array(mean_bgr,dtype=np.float32)
        #
        # Limage0 = cv2.cvtColor(Limage0, cv2.COLOR_BGR2RGB)
        # Limage1 = cv2.cvtColor(Limage1, cv2.COLOR_BGR2RGB)
        # Limage2 = cv2.cvtColor(Limage2, cv2.COLOR_BGR2RGB)
        # Limage3 = cv2.cvtColor(Limage3, cv2.COLOR_BGR2RGB)

        Limage = cv2.cvtColor(Limage, cv2.COLOR_BGR2RGB)

        Limage = self.transform(Limage)

        # Limage0 = self.transform(Limage0)
        # Limage1 = self.transform(Limage1)
        # Limage2 = self.transform(Limage2)
        # Limage3 = self.transform(Limage3)
        # print(Limage)

        name =name.replace('d', 'gd')
        Rimage = cv2.imread(f"./size_512/gall-test/{name}")
        # convert it from BGR to RGB
        Rimage = cv2.cvtColor(Rimage, cv2.COLOR_BGR2RGB)
        Rimage = self.transform(Rimage)

        # targets = int(name[2])
        # #  [1, 0, 0, 0]  [0, 1, 0, 0] [0, 0, 1, 0]  [0, 0, 0, 1]
        # targets = [0, 0, 1, 0]

        zidian = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        targets = zidian[int(name[2])]




        # print(self.Limage_names[index],name, targets)
        return {
            # 'limage': torch.tensor(Limage),
            # 'rimage': torch.tensor(Rimage),
            # 'label': torch.tensor(targets)
            'limage': torch.tensor(Limage, dtype=torch.float32),
            'rimage': torch.tensor(Rimage, dtype=torch.float32),
            'label': torch.tensor(targets, dtype=torch.float32)
        }


class ImageDatasetnew(Dataset):

    def __init__(self, traincsv,testcsv, train, validation, test):

        self.traincsv = traincsv
        self.testcsv = testcsv
        self.train = train
        self.validation = validation
        self.test = test
        # self.all_image_names = self.csv[:]['Id']
        # self.left_image_names = self.csv[:, 3]
        # self.right_image_names = self.csv[:, 4]
        self.left_image_names = self.traincsv[:]['Left-Fundus']
        self.right_image_names = self.traincsv[:]['Right-Fundus']
        self.tleft_image_names = self.testcsv[:]['Left-Fundus']
        self.tright_image_names = self.testcsv[:]['Right-Fundus']
        self.images = []
        # imgleft_name = self.csv[:, 3]  # 输出第一列数据
        # imgright_name = self.csv[:, 4]  # 输出第一列数据
        # self.all_labels = np.array(self.csv.drop(['Id', 'Genre'], axis=1))
        self.all_labels = np.array(self.traincsv.drop(
            ['ID', 'Patient Age', 'Patient Sex', 'Left-Fundus', 'Right-Fundus', 'Left-Diagnostic Keywords',
             'Right-Diagnostic Keywords'], axis=1))
        self.tall_labels = np.array(self.csv.drop(
            ['ID', 'Patient Age', 'Patient Sex', 'Left-Fundus', 'Right-Fundus', 'Left-Diagnostic Keywords',
             'Right-Diagnostic Keywords', '1', '2', '3', '4'], axis=1))
        self.train_len = len(self.traincsv)
        # self.valid_len = int(0.015*len(self.csv))
        self.valid_len = len(self.testcsv)

        # set the training data images and labels
        if self.train == True:
            print(f"Number of training images: {self.train_len}")

            self.Limage_names = list(self.left_image_names[:self.train_len])
            self.Rimage_names = list(self.right_image_names[:self.train_len])
            self.tLimage_names = list(self.left_image_names[:self.testcsv])
            self.tRimage_names = list(self.right_image_names[:self.testcsv])
            self.images = self.Limage_names + self.Rimage_names
            self.labels = list(self.all_labels[:self.train_len])
            self.tlabels = list(self.all_labels[:self.valid_len])

            # define the training transform
            self.transform = transforms.Compose([

                #官方
                transforms.ToPILImage(),
                transforms.Resize((512, 512)),
                transforms.RandomResizedCrop(224),
                # transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1), transforms.RandomVerticalFlip(p=1),
                transforms.ToTensor(),  # 转化为张量并归一化为[0-1]
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


            ])

        elif self.validation == True:
            print(f"Number of Validation images: {self.valid_len}")
            self.Limage_names = list(self.tleft_image_names[:self.testcsv])
            self.Rimage_names = list(self.tright_image_names[:self.testcsv])
            self.images = self.Limage_names + self.Rimage_names
            self.labels = list(self.all_labels[-self.valid_len:])


            # define the validatio transform
            self.transform = transforms.Compose([
                # transforms.ToPILImage(),
                # transforms.Resize((400, 400)),
                # transforms.ToTensor(),
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),

                # transforms.ToPILImage(),
                # transforms.Resize((300, 300)),
                # transforms.ToTensor(),
                # # transforms.RandomErasing(p=0.2),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # resnet数据集
                # transforms.Normalize([0.5820, 0.4512, 0.4023], [0.2217, 0.1858, 0.1705]),
                #官方
                transforms.ToPILImage(),
                # transforms.Resize((512, 512)),  # 随机长宽比裁剪为224*224
                transforms.Resize((224, 224)),
                # transforms.RandomHorizontalFlip(p=1), transforms.RandomVerticalFlip(p=1),
                transforms.ToTensor(),  # 转化为张量并归一化为[0-1]
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        elif self.test == True:
            print("Number of Test images: 10")
            self.Limage_names = list(self.left_image_names[-self.valid_len:-10])
            self.Rimage_names = list(self.right_image_names[-self.valid_len:-10])
            self.labels = list(self.all_labels[-10:])

            # define the test transform
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),

            ])

    def __len__(self):
        return len(self.Limage_names)
    # def __len__(self, index):
    #     return len(self.Limage_names,self.Rimage_names)

    def __getitem__(self, index):
    # def __getitem__(self, index, index1):

        name=self.Limage_names[index]

        Limage = cv2.imread(f"../Test5_resnet/OIA-ODIR/Training Set/Images/{self.Limage_names[index]}")  #这个可能有问题
        Limage = cv2.cvtColor(Limage, cv2.COLOR_BGR2RGB)
        # limage = cv2.imread("OIA-ODIR\Training Set\Images\0_left.jpg")
        # limage = cv2.imread("./OIA-ODIR/Training Set/Images/{self.Limage_names[index]}.jpg")
        # convert it from BGR to RGB

        # cv2.namedWindow("Hello1", cv2.WINDOW_AUTOSIZE)
        # cv2.imshow("Hello1", Limage)
        # Limage = cv2.cvtColor(Limage, cv2.COLOR_BGR2RGB)
        #
        # cv2.namedWindow("Hello", cv2.WINDOW_AUTOSIZE)
        # cv2.imshow("Hello", Limage)

        # limage = cv2.open(limage).convert('RGB')

    # apply transforms
        Limage = self.transform(Limage)



        name =name.replace('left', 'right')
        Rimage = cv2.imread(f"../Test5_resnet/OIA-ODIR/Training Set/Images/{name}")
        # convert it from BGR to RGB
        Rimage = cv2.cvtColor(Rimage, cv2.COLOR_BGR2RGB)
        # apply transforms
        Rimage = self.transform(Rimage)

        targets = self.labels[index]     #看出来有问题

        # print(self.Limage_names[index],name, targets)
        return {
            'limage': torch.tensor(Limage, dtype=torch.float32),
            'rimage': torch.tensor(Rimage, dtype=torch.float32),
            'label': torch.tensor(targets, dtype=torch.float32)
        }



class TestImageDataset(Dataset):

    def __init__(self, csv, test, validation):

        self.csv = csv
        self.validation = validation
        self.test = test
        self.left_image_names = self.csv[:]['Left-Fundus']
        self.right_image_names = self.csv[:]['Right-Fundus']
        self.images = []
        # imgleft_name = self.csv[:, 3]  # 输出第一列数据
        # imgright_name = self.csv[:, 4]  # 输出第一列数据
        # self.all_labels = np.array(self.csv.drop(['Id', 'Genre'], axis=1))
        # self.all_labels = pd.read_csv('../Test5_resnet/on-site test annotation (English).csv',
        #                     usecols=['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O'])
        self.all_labels = np.array(self.csv.drop(
            ['ID', 'Patient Age', 'Patient Sex', 'Left-Fundus', 'Right-Fundus', 'Left-Diagnostic Keywords',
             'Right-Diagnostic Keywords','1','2','3','4'], axis=1))

        self.test_len = int(len(self.csv))


        # set the training data images and labels
        if self.test == True:
            print(f"Number of training images: {self.test_len}")

            self.Limage_names = list(self.left_image_names[:self.test_len])
            self.Rimage_names = list(self.right_image_names[:self.test_len])
            self.images = self.Limage_names + self.Rimage_names
            self.labels = list(self.all_labels[:self.test_len])

            # define the training transform
            self.transform = transforms.Compose([
                # transforms.ToPILImage(),         # 这个一定要有 否则报错
                # transforms.Resize((400, 400)),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomRotation(degrees=45),
                # transforms.ToTensor(),
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),# 迁移学习的话需要减去 来去中心化（resnet 视频

                #  Github上jupter某人方案
                # transforms.ToPILImage(),
                # transforms.Resize((300, 300)),
                # transforms.RandomHorizontalFlip(p=1), transforms.RandomVerticalFlip(p=1),
                # transforms.RandomRotation(degrees=30),
                #
                # transforms.ToTensor(),
                # # transforms.RandomErasing(p=0.2),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  #resnet数据集

                # transforms.Normalize([0.5820, 0.4512, 0.4023], [0.2217, 0.1858, 0.1705]),

                # 官方
                # transforms.ToPILImage(),
                # # transforms.RandomResizedCrop(224),  # 随机长宽比裁剪为224*224
                # transforms.Resize((512, 512)),
                # transforms.CenterCrop(448),
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224),  # 随机长宽比裁剪为224*224
                # transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1),  # contrast=0.5, saturation=0.5, hue=0.5

                transforms.ToTensor(),  # 转化为张量并归一化为[0-1]
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            ])



    def __len__(self):
        return len(self.Limage_names)

    # def __len__(self, index):
    #     return len(self.Limage_names,self.Rimage_names)

    def __getitem__(self, index):
        # def __getitem__(self, index, index1):

        name = self.Limage_names[index]

        # Limage = cv2.imread(f"../Test5_resnet/OIA-ODIR/On-site Test Set/Images/{self.Limage_names[index]}")  # 这个可能有问题
        #
        # Limage = cv2.cvtColor(Limage, cv2.COLOR_BGR2RGB)
        Limage = Image.open(f"../Test5_resnet/OIA-ODIR/On-site Test Set/Images/{self.Limage_names[index]}").convert(
            'RGB')


        # apply transforms
        Limage = self.transform(Limage)

        name = name.replace('left', 'right')
        # Rimage = cv2.imread(f"../Test5_resnet/OIA-ODIR/On-site Test Set/Images/{name}")
        # # convert it from BGR to RGB
        # Rimage = cv2.cvtColor(Rimage, cv2.COLOR_BGR2RGB)
        Rimage = Image.open(f"../Test5_resnet/OIA-ODIR/On-site Test Set/Images/{name}").convert(
            'RGB')

        # apply transforms
        Rimage = self.transform(Rimage)


        # Limage = Limage.unsqueeze(0)  # 增加一个维度 必须要有
        # Rimage = Rimage.unsqueeze(0)  # 增加一个维度 必须要有

        targets = self.labels[index]  # 看出来有问题

        # print(self.Limage_names[index],name, targets)
        return {
            'limage': torch.tensor(Limage, dtype=torch.float32),
            'rimage': torch.tensor(Rimage, dtype=torch.float32),
            'label': torch.tensor(targets, dtype=torch.float32)
        }


class TestImageDatasetoff(Dataset):

    def __init__(self, csv, test, validation):

        self.csv = csv
        self.validation = validation
        self.test = test
        self.left_image_names = self.csv[:]['Left-Fundus']
        self.right_image_names = self.csv[:]['Right-Fundus']
        self.images = []
        # imgleft_name = self.csv[:, 3]  # 输出第一列数据
        # imgright_name = self.csv[:, 4]  # 输出第一列数据
        # self.all_labels = np.array(self.csv.drop(['Id', 'Genre'], axis=1))
        # self.all_labels = pd.read_csv('../Test5_resnet/on-site test annotation (English).csv',
        #                     usecols=['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O'])
        self.all_labels = np.array(self.csv.drop(
            ['ID', 'Patient Age', 'Patient Sex', 'Left-Fundus', 'Right-Fundus', 'Left-Diagnostic Keywords',
             'Right-Diagnostic Keywords'], axis=1))
        self.test_len = int(len(self.csv))


        # set the training data images and labels
        if self.test == True:
            print(f"Number of testing images: {self.test_len}")

            self.Limage_names = list(self.left_image_names[:self.test_len])
            self.Rimage_names = list(self.right_image_names[:self.test_len])
            self.images = self.Limage_names + self.Rimage_names
            self.labels = list(self.all_labels[:self.test_len])

            # define the training transform
            self.transform = transforms.Compose([


                # 官方
                transforms.ToPILImage(),
                # # transforms.RandomResizedCrop(224),  # 随机长宽比裁剪为224*224
                # transforms.Resize((512, 512)),
                transforms.Resize((512, 512)),
                transforms.CenterCrop(448),
                # transforms.Resize((256, 256)),
                # transforms.CenterCrop(224),

                # transforms.Resize((1024, 1024)),
                # transforms.Resize((224, 224)),
                transforms.ToTensor(),  # 转化为张量并归一化为[0-1]
                # transforms.Normalize([0.340, 0.214, 0.115], [0.352, 0.239, 0.154])
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # transforms.Normalize([0.5820, 0.4512, 0.4023], [0.2217, 0.1858, 0.1705])

            ])



    def __len__(self):
        return len(self.Limage_names)

    # def __len__(self, index):
    #     return len(self.Limage_names,self.Rimage_names)

    def __getitem__(self, index):
        # def __getitem__(self, index, index1):

        name = self.Limage_names[index]

        Limage = cv2.imread(f"../Test5_resnet/OIA-ODIR/Off-site Test Set/Images/{self.Limage_names[index]}")  # 这个可能有问题

        Limage = cv2.cvtColor(Limage, cv2.COLOR_BGR2RGB)
        # Limage = Image.open(f"../Test5_resnet/OIA-ODIR/Off-site Test Set/Images/{self.Limage_names[index]}").convert(
        #     'RGB')

        # plt.imshow(Limage)
        # plt.show()


        # apply transforms
        Limage = self.transform(Limage)

        name = name.replace('left', 'right')
        Rimage = cv2.imread(f"../Test5_resnet/OIA-ODIR/Off-site Test Set/Images/{name}")
        # convert it from BGR to RGB
        Rimage = cv2.cvtColor(Rimage, cv2.COLOR_BGR2RGB)
        # Rimage = Image.open(f"../Test5_resnet/OIA-ODIR/Off-site Test Set/Images/{name}").convert('RGB')

        # apply transforms
        Rimage = self.transform(Rimage)


        # Limage = Limage.unsqueeze(0)  # 增加一个维度 必须要有
        # Rimage = Rimage.unsqueeze(0)  # 增加一个维度 必须要有

        targets = self.labels[index]  # 看出来有问题

        # print(self.Limage_names[index],name, targets)
        return {
            'limage': torch.tensor(Limage),
            'rimage': torch.tensor(Rimage),
            'label': torch.tensor(targets)
            # 'limage': torch.tensor(Limage, dtype=torch.float32),
            # 'rimage': torch.tensor(Rimage, dtype=torch.float32),
            # 'label': torch.tensor(targets, dtype=torch.float32)
        }
