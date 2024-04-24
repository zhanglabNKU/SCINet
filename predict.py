import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from dataset import ImageDataset
import pandas as pd
from torchvision import models
import numpy as np
from dataset import TestImageDataset
from model import vggme
from model import DENetvgg
from model import DENetresnet
from model import MyVggModel
import torch, glob, cv2
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np


def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            }


def kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)
    return (po - pe) / (1 - pe)



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #vgg
    # model0 = models.vgg16()
    # model1 = models.vgg16()
    # net = DENetvgg(model0, model1)

    #resnet
    model0 = models.resnext50_32x4d()
    model1 =models.resnext50_32x4d()
    net = DENetresnet(model0, model1)

    # create model

    # load model weights
    # weights_path = "outputs/modelvggbest29.pth"
    weights_path = "outputs/modelrestnextbest1.pth"
    # model0.load_state_dict(torch.load(weights_path, map_location=device))
    # model1.load_state_dict(torch.load(weights_path, map_location=device))
    net.load_state_dict(torch.load(weights_path, map_location=device))
    net.to(device)

    csvname='../Test5_resnet/on-site test annotation (English).csv'
    # test_csv = pd.read_csv('../Test5_resnet/on-site test annotation (English).csv')
    test_csv = pd.read_csv(csvname)
    # test_data = ImageDataset(test_csv, train=True, validation=False, test=False)
    left_image_names = test_csv[:]['Left-Fundus']
    right_image_names = test_csv[:]['Right-Fundus']
    label = pd.read_csv(csvname,
                        usecols=['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O'])
    # label = pd.read_csv('../Test5_resnet/on-site test annotation (English).csv', usecols=['N','D','G','C','A','H','M','O'])

    # label = test_csv[:]['N','D','G','C','A','H','M','O']
    # label = np.array(test_csv.drop(
    #         ['ID', 'Patient Age', 'Patient Sex', 'Left-Fundus', 'Right-Fundus', 'Left-Diagnostic Keywords',
    #          'Right-Diagnostic Keywords'], axis=1))
    labels = list(label[:len(label)])
    # torch.no_grad()
    imgfile = glob.glob(r"OIA-ODIR\On-site Test Set\Images")  # 输入要预测的图片所在路径
    print(len(left_image_names), label)
    net.eval()
    with torch.no_grad():
     for i in left_image_names:
        imgfile1 = i.replace("\\", "/")
        imgfile2 = i.replace('left', 'right')
        # img = cv2.imdecode(np.fromfile(imgfile1, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        # img = cv2.resize(img, (64, 64))  # 是否需要resize取决于新图片格式与训练时的是否一致
        # img = cv2.imdecode(np.fromfile(imgfile1, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(f"../Test5_resnet/OIA-ODIR/On-site Test Set/Images/{imgfile1}")
        # print(imgfile1,labels[i])
        print(imgfile1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(f"../Test5_resnet/OIA-ODIR/On-site Test Set/Images/{imgfile2}")
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomResizedCrop(224),  # 随机长宽比裁剪为224*224
            transforms.ToTensor(),  # 转化为张量并归一化为[0-1]
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        model_result = []
        AUROCs = []
        targets = []
        img1 = transform(img1)
        img1 = img1.unsqueeze(0)  # 增加一个维度 必须要有
        img1 = img1.to(device)
        img2 = transform(img2)
        img2 = img2.unsqueeze(0)  # 增加一个维度 必须要有
        img2 = img2.to(device)
        # img2 = img2.squeeze(0)
        output = torch.squeeze(net(img1, img2))

        # predict = torch.softmax(output, dim=0)
        predict = torch.sigmoid(output)  # 不加就爆炸  再取最大两个
        print(predict)
        model_result.extend(predict.cpu().numpy())

    # targets.extend(label.cpu().numpy())
    # result = calculate_metrics(np.array(model_result), np.array(targets))
    # print(
    #     "micro f1: {:.3f} "
    #     "macro f1: {:.3f} "
    #     "samples f1: {:.3f}".format(
    #         result['micro/f1'],
    #         result['macro/f1'],
    #         result['samples/f1']))
        # predict_cla = torch.argmax(predict).numpy()





    # result = calculate_metrics(np.array(predict), np.array(labels))
    # print("micro f1: {:.3f} ""macro f1: {:.3f} ""samples f1: {:.3f}".format(result['micro/f1'],
    #                                                                         result['macro/f1'],
    #                                                                         result['samples/f1']))
    #
    # #ROC
    # fpr, tpr, thresholds = metrics.roc_curve(output, labels)
    # print('FPR:', fpr)
    # print('TPR:', tpr)
    # print('thresholds:', thresholds)
    # plt.plot(fpr, tpr)
    # plt.show()

    # #kappa
    # matrix = np.array(matrix) #混淆矩阵
    # print(kappa(matrix))


if __name__ == '__main__':
    main()
