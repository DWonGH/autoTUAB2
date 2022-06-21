import csv
import pandas as pd
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import init
import numpy as np
from sklearn.model_selection import train_test_split

n_repetition=5
length=10
use_his=False
adap_pool=False
use_hybrid=True
threshold=0.5
total=800

# with open('./training_detail.csv','r') as f:
filename='./training_detail_1.csv'
# total = sum(1 for line in open(filename))
# print('The total lines is ',total)

# with open('./training_detail.csv', newline='') as csvfile:
#     results = csv.reader(csvfile, delimiter=',')
#     print(results[1])
results= pd.read_csv(filename)
print(results)
# print(results.loc[2,'16000'])
labels=[]
valid_lens=[]
data=[]
pd_valid_lens=results.loc[2,:].tolist()[:total+1]
for i in range(len(pd_valid_lens)):
    pd_valid_lens[i]=int(pd_valid_lens[i])
pd_data=results.loc[1,:].tolist()
for i in range(len(pd_data)):
    pd_data[i]=float(pd_data[i])
pd_labels=results.loc[0,:].tolist()
for i in range(len(pd_labels)):
    if pd_labels[i]=='FALSE':
        pd_labels[i]=0
    else:
        pd_labels[i] = 1

def creat_his(raw,length=10):
    his=np.zeros(length)
    for i in raw:
        his[int(i// (1 / length + 0.001))] += 1
    return his


for i in range(total):
    # if (pd_valid_lens[i+1]-pd_valid_lens[i])!=20:
    #     total=total-1
    #     continue
    valid_lens.append(pd_valid_lens[i+1]-pd_valid_lens[i])
    labels.append(pd_labels[pd_valid_lens[i]])
    if use_his:
        if use_hybrid:
            data.append(np.concatenate([creat_his(pd_data[pd_valid_lens[i]:pd_valid_lens[i+1]],length=length),pd_data[pd_valid_lens[i]:pd_valid_lens[i+1]]+[0]*(20-valid_lens[-1])],axis=0)  )
        else:
            data.append(creat_his(pd_data[pd_valid_lens[i]:pd_valid_lens[i+1]],length=length))
    else:
        data.append(pd_data[pd_valid_lens[i]:pd_valid_lens[i+1]]+[0]*(20-valid_lens[-1]))

print('total',total)
# print(labels)
# print(valid_lens)
# print(data)
# print(len(labels))
# print(len(valid_lens))
# print(len(data))
# print('min',min(valid_lens))
class MyDataset(Dataset):
    def __init__(self,data,labels,valid_lens):
        self.valid_lens=valid_lens
        self.labels=labels
        self.data=torch.tensor(data,dtype=torch.float)
    def __len__(self):
        return len(self.data)
	#打开index对应图片进行预处理后return回处理后的图片和标签
    def __getitem__(self, index):
        return self.data[index],self.labels[index],self.valid_lens[index]
#实例化对象
dataset = MyDataset(data,labels,valid_lens)
ave_test_acc=0
ave_ori_acc=0
ave_argmax_acc=0
ave_mean_acc=0
for n_time in range(n_repetition):
    idx_train, idx_test = train_test_split(np.arange(total),
                                                 random_state=n_time,
                                                 train_size=0.8,
                                                 shuffle=True)
    test_set = torch.utils.data.Subset(dataset,idx_test)
    train_set = torch.utils.data.Subset(dataset, idx_train)
    #将数据集导入DataLoader，进行shuffle以及选取batch_size
    data_loader = DataLoader(train_set,batch_size=1,shuffle=True,num_workers=0)

    # for datum,label,valid_len in data_loader:
    #     print('datum',datum)
    #     print('label',label)
    #     print('valid_len',valid_len)

    class decision_model(nn.Module):
        def __init__(self,adap_pool=True):
            super().__init__()
            self.adap_pool=adap_pool
            if adap_pool:
                self.pooling=nn.AdaptiveAvgPool1d((10))
                self.classifer=nn.Linear(10,2)
            else:
                self.classifer=nn.Linear(20,2)
            self.log_softmax = nn.LogSoftmax(dim=1)
            init.normal_(self.classifer.weight, 0, 0.01)
        def forward(self,x,valid_len):
            if self.adap_pool:
                x=x[:,:valid_len]
                x=self.pooling(x)
            x=self.classifer(x)
            x=self.log_softmax(x)
            return x
    class histogram_model(nn.Module):
        def __init__(self,length=10,use_hybrid=False):
            super().__init__()
            self.length=length
            if use_hybrid:
                self.classifer = nn.Linear(self.length+20, 2)
            else:
                self.classifer=nn.Linear(self.length,2)
            self.log_softmax = nn.LogSoftmax(dim=1)
            init.normal_(self.classifer.weight, 0, 0.01)
        def forward(self,x,valid_len):
            x=self.classifer(x)
            x=self.log_softmax(x)
            return x

    if use_his:
        model=histogram_model(length=length,use_hybrid=use_hybrid)
    else:
        model=decision_model(adap_pool=adap_pool)

    # datum1,label1,valid_len1=next(iter(data_loader))
    # print(model(datum1))
    cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
    device = 'cuda' if cuda else 'cpu'
    model.to(device)
    lr=0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.nn.NLLLoss()
    model.train()
    num_epochs=20

    for epoch in range(num_epochs):
        all=0
        right=0

        for batch in data_loader:
            optimizer.zero_grad()
            X, Y,valid_len = [x.to(device) for x in batch]

            # print('x', X.shape)
            # print('y', Y.shape)
            Y_hat= model(X, valid_len)
            hat_label=torch.argmax(Y_hat,-1)
            res=sum(hat_label== Y)
            all+=len(Y)
            right+=res
            # print('hat', Y_hat.shape)
            l = loss(Y_hat, Y)
            l.backward()  # Make the loss scalar for `backward`
            optimizer.step()
        if (epoch+1)%2==0:
            print('epoch:',epoch+1)
            print('loss:',l)
            print('acc',right/all)
    model.eval()
    all = 0
    right = 0
    all_ori = 0
    right_ori = 0
    right_argmax=0
    all_argmax=0
    right_mean=0
    all_mean=0
    data_loader = DataLoader(test_set,batch_size=1,shuffle=False,num_workers=0)
    for batch in data_loader:
        X, Y, valid_len = [x.to(device) for x in batch]
        compare_res=torch.sum((X > 0.5)==Y,dim=-1)
        right_ori+=sum(compare_res)
        all_ori += X.numel()
        for i in compare_res:
            if i>10:
                right_argmax+=1
            all_argmax+=1
        mean_res=torch.mean(X,dim=-1)
        for i,j in zip(mean_res,Y):
            if (i>0.5)==j:
                right_mean+=1
            all_mean+=1

        Y_hat = model(X, valid_len)
        hat_label = torch.argmax(Y_hat, -1)
        res = sum(hat_label == Y)
        all += len(Y)
        right += res
    print('test_acc', right / all)
    print('ori_acc',right_ori/all_ori)
    print('argmax_acc',right_argmax/all_argmax)
    print('mean_acc',right_mean/all_mean)
    ave_test_acc+=right / all
    ave_ori_acc+=right_ori/all_ori
    ave_argmax_acc+=right_argmax/all_argmax
    ave_mean_acc+=right_mean/all_mean

ave_test_acc=float(ave_test_acc/n_repetition)
ave_ori_acc=float(ave_ori_acc/n_repetition)
ave_argmax_acc=float(ave_argmax_acc/n_repetition)
ave_mean_acc=float(ave_mean_acc/n_repetition)
print('ave_test_acc',ave_test_acc)
print('ave_ori_acc',ave_ori_acc)
print('ave_argmax_acc',ave_argmax_acc)
print('ave_mean_acc',ave_mean_acc)
with open('./decision_result.csv','a') as f:
    writer=csv.writer(f, delimiter=',',lineterminator='\n',)
    writer.writerow(['use_hybrid','n_repetition','length','use_his','adap_pool','ave_test_acc','ave_ori_acc','ave_argmax_acc','ave_mean_acc'])
    writer.writerow([use_hybrid,n_repetition,length,use_his,adap_pool,ave_test_acc,ave_ori_acc,ave_argmax_acc,ave_mean_acc])













