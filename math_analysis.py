import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df1=pd.read_csv('temperature_dataset.txt',header=None,sep='\s+') #读取文件
df1.columns=['体温','性别','心率'] #设置列索引

for i in list(df1.columns):
   # 获取各个指标的最大值和最小值
    Max = np.max(df1[i])
    Min = np.min(df1[i])
    df1[i] = (df1[i] - Min)/(Max - Min)
print(df1)


#print(df1) # 将文件打印
#
# #fig=plt.figure(figsize=(16,5)) #设置画布
# df2=df1[df1['性别']==1] #选择性别为1的行，注意”]“的位置
# df2=df2.reset_index(drop=True) #将文件第一行索引去掉
# plt.scatter(df2.index,df2['体温'],s=40,c='r',label='male') #c==颜色,s==点个数
# plt.legend() #根据label产生图例
#
# df3=df1[df1['性别']==2]
# df3=df3.reset_index(drop=True)
# plt.scatter(df3.index,df3['体温'],s=40,c='b',label='female')
# plt.legend()
#
# plt.ylabel("TM")
# plt.xlabel("index")
# plt.grid() #显示网格线
#
# fig2=plt.figure(figsize=(10,5))
# df_tempreture=df1['体温']
# # 绘制体温直方图
# df_tempreture.hist(bins=20,alpha = 0.5)
# # 密度图也被称作KDE图,调用plt时加上kind='kde'即可生成一张密度图。
# df_tempreture.plot(kind = 'kde', secondary_y=True)
#
# plt.show()