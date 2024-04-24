import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from torchvision import models
import numpy as np
from dataset import ImageDataset,ImageDatasettest,TestImageDataset
from thop import profile
from model3 import DENetattention3,TSNet
from thop import clever_format
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.metrics import roc_auc_score,cohen_kappa_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
from timm import create_model



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True




def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1-epoch/num_epochs)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr






def calculate_metrics(pred, target, threshold=0.5):
    pred = pred.flatten()
    target = target.flatten()
    kappa = cohen_kappa_score(target, pred>threshold)
    f1 = f1_score(target,  pred>threshold, average='macro')
    auc = roc_auc_score(target, pred)
    final_score = (kappa + f1 + auc) / 3.0
    return kappa, f1, auc, final_score


def calculate_metricsf1(pred, target, threshold=0.1):
    pred = np.array(pred > threshold, dtype=float)
    pred = pred.flatten()
    target = target.flatten()
    kappa = cohen_kappa_score(target, pred)
    f1_micro = f1_score(target, pred, average='macro')
    # f1_macro = f1_score(target, pred, average='f1_macro')
    # f1_samples = f1_score(target, pred, average='samples')
    auc = roc_auc_score(target, pred)
    final_score = (kappa + f1_micro + auc) / 3.0
    return kappa, f1_micro, auc, final_score

    # kappa, f1, auc, final_score = ODIR_Metrics(gt_data[:,1:], pr_data[:,1:])
    # print("kappa score:", kappa, " f-1 score:", f1, " AUC vlaue:", auc, " Final Score:", final_score)


def calculate_metricsf2(pred, target, threshold=0.2):
    pred = np.array(pred > threshold, dtype=float)
    pred = pred.flatten()
    target = target.flatten()
    kappa = cohen_kappa_score(target, pred)
    f1_micro = f1_score(target, pred, average='macro')
    # f1_macro = f1_score(target, pred, average='f1_macro')
    # f1_samples = f1_score(target, pred, average='samples')
    auc = roc_auc_score(target, pred)
    final_score = (kappa + f1_micro + auc) / 3.0
    return kappa, f1_micro, auc, final_score


def calculate_metricsf3(pred, target, threshold=0.4):
    pred = pred.flatten()
    target = target.flatten()
    kappa = cohen_kappa_score(target, pred > threshold)
    f1_micro = f1_score(target, pred > 0.6, average='macro')
    # f1_macro = f1_score(target, pred, average='f1_macro')
    # f1_samples = f1_score(target, pred, average='samples')
    auc = roc_auc_score(target, pred)
    final_score = (kappa + f1_micro + auc) / 3.0
    return kappa, f1_micro, auc, final_score


def calculate_metricsf4(pred, target, threshold=0.4):
    pred = pred.flatten()
    target = target.flatten()
    kappa = cohen_kappa_score(target, pred > threshold)
    f1_micro = f1_score(target, pred > 0.5, average='macro')
    # f1_macro = f1_score(target, pred, average='f1_macro')
    # f1_samples = f1_score(target, pred, average='samples')
    auc = roc_auc_score(target, pred)
    final_score = (kappa + f1_micro + auc) / 3.0
    return kappa, f1_micro, auc, final_score

# def multi_class_auc(all_target, all_output, num_c = None):
#
#     all_output = np.stack(all_output)
#     all_target = label_binarize(all_target, classes=list(range(0, num_c)))
#     auc_sum = []
#
#     for num_class in range(0, num_c):
#         try:
#             auc = roc_auc_score(all_target[:, num_class], all_output[:, num_class])
#             auc_sum.append(auc)
#         except ValueError:
#             pass
#
#     auc = sum(auc_sum) / float(len(auc_sum))
#
#     return auc
# def compute_AUCs(gt, pred):
#
#
#     AUROCs = []
#     for i in range(8):
#         AUROCs.append(roc_auc_score(gt[:, i], pred[:, i]))
#     return AUROCs




def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    warnings.filterwarnings("ignore")
    setup_seed(16)#16


    # 3 resnet18    optimizer = optim.SGD(net.parameters(), lr=0.0001)   #0.0001 yyds
    #model0 = models.resnet18(pretrained=True)
    #model1 = models.resnet18(pretrained=True)
    # net = TSNet(model0, model1)  # DENetresnet

    # 4 resnet50
    #model0 = models.resnet50(pretrained=True)
    #model1 = models.resnet50(pretrained=True)
    #model0 = models.resnet34(pretrained=True)
    #model1 = models.resnet34(pretrained=True)
    # net = DENetresnet(model0, model1)  # DENetresnet
    config = {
        'img_size': 512,  # 修改输入图像的尺寸为512x512
        'in_chans': 3,  # 输入图像的通道数，默认为3
    }
    model0 = create_model('vit_large_patch16_224', pretrained=True, num_classes=1,**config)
    model0.head = nn.Identity()
    model1 = create_model('vit_large_patch16_224', pretrained=True, num_classes=1,**config)
    model1.head = nn.Identity()

    net = DENetattention3(model0,model1)   #DENetresnet TSNet
    input1 = torch.randn(4, 3, 512, 512)  # 根据第一个输入的维度调整
    input2 = torch.randn(4, 3, 512, 512)  # 根据第二个输入的维度调整

    inputs = (input1, input2)
    macs, params = profile(net, inputs=inputs)
    macs, params = clever_format([macs, params], "%.3f")
    print(f"模型的参数数量: {params}")
    print(f"模型的FLOPs: {macs}")



    net.to(device)
    # net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # for param in net.parameters():  #冻结参数
    #     param.requires_grad = False

                                    #MultiLabelMarginLoss
    # loss_function = nn.BCELoss()  #二阶交叉熵损失函数  多标签统一   binary_crossentropy  推荐用sigmoid做输出层激活
    loss_function= nn.BCEWithLogitsLoss() #不需要sigmoid
    # loss_function  = FocalLoss(num_class=8, smooth=0.1)
    # loss_function = FocalLoss_MultiLabel()

    # loss_function = nn.MultiLabelMarginLoss()
    # loss_function = nn.MultiLabelSoftMarginLoss()  #验证需要加sigmoid
    # loss_function = nn.CrossEntropyLoss()  #  output_C = torch.softmax(output_C,1)  # 不加就


    # optimizer = optim.SGD(net.parameters(), lr=0.0003)   #0.0001 yyds
    optimizer = optim.SGD(net.parameters(), lr=0.002,momentum=0.9,weight_decay=5e-4)  #3e-4  5e-4  20个epoch  adam
    # optimizer = optim.Adam(net.parameters(), lr=0.005,eps=1e-8,weight_decay=1e-4)
    # optimizer = optim.Adam(net.parameters(), lr=0.0002,eps=1e-8,weight_decay=1e-4)
    # # lroptimizer = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    # lroptimizer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=0)
    #(bs 4  lr 0.00006)
    batch_size = 4   #6  #res18-12                              #optim.Adam(params, lr=0.0002, weight_decay=0.9)
    lr_list = []
    epochs = 300


    # train dataset
    train_data = ImageDataset(train=True, validation=False, test=False)

    # validation dataset
    test_data = ImageDatasettest(train=True, validation=False, test=False)


    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # validation data loader

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    # starting the training loop and validation
    train_loss = []
    valid_loss = []
    val_num = len(test_data)

    # print("using {} images for training, {} images for validation.".format(train_data,
    #                                                                        test_data))


    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        # lroptimizer.step()  ####################################################################这个不要忘了  如果用余弦之类
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        if epoch <30 :
            for p in optimizer.param_groups:  # 可以不用了
                # p['lr'] *= 0.9
                p['lr'] = 0.002 * (1 - epoch / 30) ** 0.9 #0.006
            lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        else:
            for p in optimizer.param_groups:  # 可以不用了
                # p['lr'] *= 0.95
                p['lr'] = 0.000105 * 0.95
            lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
       

        for i, data in enumerate(train_bar): #可用但爆内存版
            images1,images2, labels = data['limage'].to(device),data['rimage'].to(device), data['label'].to(device)  #注意  这里是limage！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            optimizer.zero_grad()
            outputs = net(images1,images2)


            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # if  (epoch+1) % 2 == 0:
        if epoch+1 >= 0 :

            # validate
            net.eval()
            acc = 0.0  # accumulate accurate number / epoch

            with torch.no_grad():
                model_result = []

                targets = []
                # val_bar = tqdm(train_loader)
                val_bar = tqdm(test_loader)

                for i, data in enumerate(val_bar):  # 可用但爆内存版
                # for data in val_bar:  #这个也可以
                    images1,images2, labels = data['limage'].to(device),data['rimage'].to(device), data['label'].to(device)  #注意  这里是limage！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
                    outputs = net(images1,images2)
                    outputs = torch.sigmoid(outputs)
                    # outputs = torch.softmax(outputs)

                    model_result.extend(outputs.cpu().numpy())
                    targets.extend(labels.cpu().numpy())


                        # roc +=roc_auc_score(labels, outputs)
                #f1
                # result = calculate_metricsf1(np.array(model_result), np.array(targets))
                kappa, f1, auc, final_score = calculate_metrics(np.array(model_result), np.array(targets))
                print("kappa score:", kappa, " f1 score:", f1, " AUC vlaue:", auc, " Final Score:", final_score)
                # 0.1
                kappa, f1_micro, auc, final_score = calculate_metricsf1(np.array(model_result), np.array(targets))
                print("kappa score:", kappa, " f1 score:", f1_micro, " AUC vlaue:", auc, " Final Score:", final_score)
                # 0.2
                kappa, f1_micro, auc, final_score = calculate_metricsf2(np.array(model_result), np.array(targets))
                print("kappa score:", kappa, " f1 score:", f1_micro, " AUC vlaue:", auc, " Final Score:", final_score)
                # 0.3
                kappa, f1_micro, auc, final_score = calculate_metricsf3(np.array(model_result), np.array(targets))
                print("kappa score:", kappa, " f1 score:", f1_micro, " AUC vlaue:", auc, " Final Score:", final_score)
                # 0.4
                kappa, f1_micro, auc, final_score = calculate_metricsf4(np.array(model_result), np.array(targets))
                print("kappa score:", kappa, " f1 score:", f1_micro, " AUC vlaue:", auc, " Final Score:", final_score)

                #求最大f1并保存
                # predict_y = np.array(model_result)
                # predict_y = torch.where(predict_y > 0.2,
                #                         torch.ones_like(predict_y, dtype=torch.float32),
                #                         torch.zeros_like(predict_y, dtype=torch.float32))
                # # model_result = np.array(model_result > 0.2, dtype=float)
                # f1 =f1_score(y_true=np.array(targets), y_pred=predict_y, average='micro')
                # if f1 > best_f1:
                #    best_f1 =f1
                # torch.save(net.state_dict(), 'outputs/modelresnetbest'+ str(best_f1) + '.pth')
                # print(f1)
                # torch.save(net.state_dict(), 'outputs/2resnet18n-'+ str(epoch) + '.pth')
                #AUROC
                # AUROCs=roc/val_num
                # print( 'AUROC: %.3f' %(AUROCs))

                val_accurate = acc / val_num
                # print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                #       (epoch + 1, running_loss / train_steps, val_accurate))
                print('[epoch %d] train_loss: %.3f ' %
                      (epoch + 1, running_loss / train_steps))

                newloss =running_loss / train_steps
                torch.save(net.state_dict(), 'outputs/resnet50-' + str(epoch+1) + '.pth')

                # if newloss < bestloss:
                #     bestloss = newloss
    #                 torch.save(net.state_dict(), 'outputs/2resnetn' + str(newloss) + '.pth')
    torch.save(net.state_dict(), 'outputs/resnet50-' + str(epoch+1) + '.pth')
    print('Finished Training')


if __name__ == '__main__':
    main()
