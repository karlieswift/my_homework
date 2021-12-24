"""
 Env: /anaconda3/python3.7
 Time: 2021/12/2 20:20
 Author: karlieswfit
 File: vgg16_EnglishFnt.py
 Describe:vgg16迁移学习训练数据EnglishFnt
题目：使用卷积神经元网络CNN，对多种字体的26个大写英文字母进行识别。
数据集介绍：
1- 数据集来源于Chars74K dataset，本项目选用数据集EnglishFnt中的一部分。Chars74K dataset网址链接 http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
2- A-Z共26种英文字母，每种字母对应一个文件夹（Sample011对应字母A, Sample012对应字母B,…, Sample036对应字母Z）
3- Sample011到Sample036每个文件夹下相同字母不同字体的图片约1000张，PNG格式
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
from torchsummary import  summary
from torchvision.models import vgg16
from sklearn.model_selection import train_test_split
# import pandas as pd
import os

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class vgg16_EnglishFnt(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg16 = vgg16(pretrained=True)  # 第一次运行需要下载一段时间
        # 构造自己的全连接 来替换vgg16的全连接层
        my_classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=200, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=200, out_features=26)
        )
        # 替换全连接层
        self.vgg16.classifier = my_classifier
        # EnglishFnt字母案例 的单通道 vgg16的第一层通道为3 这里需要修改为1
        self.vgg16.features[0] = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        # 先冻结vgg16卷积层的所有参数
        for param in self.vgg16.parameters():
            param.requires_grad = False
        # 更新卷积层的某几层
        for i in [28,29,30]:
            for param in self.vgg16.features[i].parameters():
                param.requires_grad=True

        # 更新全连接层
        for param in self.vgg16.classifier.parameters():
            param.requires_grad = True

    def forward(self, input):
        output = self.vgg16(input)
        return output
path='./project2/'
batch_size=64
transforms=torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(1),  #把通道改为1
    torchvision.transforms.ToTensor()
])
data_set=torchvision.datasets.ImageFolder(path,transform=transforms)
# 数据集划分
train_data,test_data=train_test_split(data_set,random_state=666,shuffle=True,test_size=0.2)
train_data_loader=torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
test_data_loader=torch.utils.data.DataLoader(dataset=test_data,batch_size=batch_size,shuffle=True)

def test(data_loader,model):
    sum = 0
    for index, (input, target) in enumerate(data_loader):
        # GPU
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        y = torch.max(output, dim=1)[1]
        sum += (y == target).sum()
    acc = sum/len(data_loader)/batch_size
#     print("acc:", acc.item())
    return acc.item()


model = vgg16_EnglishFnt().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
crossEntropyLoss = nn.CrossEntropyLoss()
# 模型加载
if os.path.exists('./EnglishFnt_vgg_model/model.pkl'):
    model.load_state_dict(torch.load('./EnglishFnt_vgg_model/model.pkl'))
    optimizer.load_state_dict(torch.load('./EnglishFnt_vgg_model/optimizer.pkl'))


def train(epoch, data_loader):
    loss_list = []
    acc_list = []
    for index, (input, target) in enumerate(data_loader):
        # GPU
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        loss = crossEntropyLoss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 模型保存
        if index % 100 == 0:
            print("epoch:{} loss:{:.4f} [{}/{}]".format(epoch + 1, loss.item(), index * batch_size,
                                                                  len(data_set)))


    torch.save(model.state_dict(), './EnglishFnt_vgg_model/model.pkl')
    torch.save(optimizer.state_dict(), './EnglishFnt_vgg_model/optimizer.pkl')

    return loss_list, acc_list


def test_all(data_loader):
    model = vgg16_EnglishFnt().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if os.path.exists('./EnglishFnt_vgg_model/model.pkl'):
        model.load_state_dict(torch.load('./EnglishFnt_vgg_model/model.pkl'))
        optimizer.load_state_dict(torch.load('./EnglishFnt_vgg_model/optimizer.pkl'))

    sum = 0
    for index, (input, target) in enumerate(data_loader):
        # GPU
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        y = torch.max(output, dim=1)[1]
        sum += (y == target).sum()
    acc = sum / len(data_loader) / batch_size
    print("acc:", acc.item())
    return acc.item()


if __name__ == '__main__':
    # for i in range(1):
    #     train(i,train_data_loader)
    test_all(test_data_loader)