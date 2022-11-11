import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

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
test_data=torch.tensor(np.array(data.tail(b)),dtype=torch.float) #x选择末尾20%,将DataFrame数据转为数组
test_label=torch.tensor(np.array(label.tail(b)),dtype=torch.float)



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

model=torch.load('model.pth')
model.eval()
acc=0
total=0
for idx,(x,label) in enumerate(test_dataset): #迭代数据集
    target=model(x)
    total+=1
    output=0
    if target >0.5 and target<1:
        output=1
    elif target <0.5 and target >0:
        pass
    if output==label:
        acc+=1
        print(True)
    else:
        print(False)
print('在测试集的准确率为：{}'.format(acc/total))