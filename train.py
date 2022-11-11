import torch
from torch.utils.data import Dataset,DataLoader
from torch import nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
np.random.seed(2022)
#定义Getdata类,并重写Dataset方法
class Getdata(Dataset):
    #初始化数据
    def __init__(self,path,label):
        self.data=path
        self.label=label
    #根据index获取文件
    def __getitem__(self, index):
        data=self.data[index]
        label=self.label[index]
        return data,label
    #获取数据的数量
    def __len__(self):
        return len(self.data)

file=pd.read_csv('temperature_dataset.txt',header=None,sep='\s+') #读取文件
#归一化操作
for i in list(file.columns):
   # 获取各个指标的最大值和最小值
    Max = np.max(file[i])
    Min = np.min(file[i])
    file[i] = (file[i] - Min)/(Max - Min)

file=file.sample(frac=1) #打乱文件顺序，frac是返回数据的比例



label=file.iloc[:,1] #选择第二列
data=file.iloc[:,[0,2]] #选择一、三列

#print(label)
a=int(len(file)*0.8) #文件的80%
train_data=torch.tensor(np.array(data.head(a)),dtype=torch.float) #选择数据前80%，将DataFrame数据转为数组
train_label=torch.tensor(np.array(label.head(a)),dtype=torch.float).reshape(104,1)
#print(train_data,train_label)

b=int(len(file)*0.2) # 文件的20%
test_data=torch.tensor(np.array(data.tail(b))) #x选择末尾20%,将DataFrame数据转为数组
test_label=torch.tensor(np.array(label.tail(b)))



train_dataset=Getdata(train_data,train_label) #调用torch的Dataset
test_dataset=Getdata(test_data,test_label)

#构建数据加载器
train_dataloader=DataLoader(train_dataset,
                            batch_size=2,
                            shuffle=True,
                            drop_last=True,
                            num_workers=0)

test_dataloader=DataLoader(test_dataset,
                            batch_size=2,
                            shuffle=False,
                            drop_last=True,
                            num_workers=0)

#定义网络，继承nn.Module
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.hiden=nn.Sequential(
            nn.Linear(2,7), #线性层将2future映射成7future
            nn.ReLU(),
            nn.Linear(7,1)
        )
    #定义前向网络
    def forward(self,x):
        x=torch.sigmoid(self.hiden(x))
        return x

model=Net() #创建网络Net


optimizer=torch.optim.Adam(params=model.parameters(),lr=0.01) #Adam优化器
loss_fn=torch.nn.MSELoss() #交叉熵损失
epoch=100 #迭代的次数


# for idx,[x,label] in enumerate(train_dataset): #迭代数据集
#     print(x,label)
#进行迭代
min_loss=1
for i in range(epoch):
    for idx,(x,label) in enumerate(train_dataset): #迭代数据集
        #print(idx)
        output=model(x)#将数据丢进模型
        loss=loss_fn(output,label) #计算损失
        optimizer.zero_grad() #梯度清零
        loss.backward() #反向求导
        optimizer.step() #权值更新
        if min_loss>loss.item():
            min_loss=loss.item()
            torch.save(model, "model.pth")
print(min_loss)




