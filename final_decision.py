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
import copy
from sklearn.model_selection import train_test_split
from util import *
from itertools import product

N_REPETITION=[5]
LENGTH=[10]
USE_HIS=[True]
ADAP_POOL=[False]
USE_HYBRID=[True]
TOTAL=[5972]
USE_SESSION_OR_PATIENTS=[None]#'patients','sessions',None
filename = './training_detail_tueg_tuab.csv'
for (n_repetition,length,use_his,adap_pool,use_hybrid,total,use_session_or_patients) in product(N_REPETITION,\
    LENGTH,USE_HIS,ADAP_POOL,USE_HYBRID,TOTAL,USE_SESSION_OR_PATIENTS):
    # n_repetition=5
    # length=7
    # use_his=True
    # adap_pool=True
    # use_hybrid=False
    # total=5972
    # use_session_or_patients='patients'

    # with open('./training_detail.csv','r') as f:

    # total = sum(1 for line in open(filename))
    # print('The total lines is ',total)
    pd_labels=[]
    pd_valid_lens=[]
    pd_data=[]
    patients=[]
    sessions=[]
    data1_len=0
    with open(filename, newline='') as csvfile:
        results = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(results):
            if i >=35 and i<39:
                pd_labels+=row
            elif i>=39 and i<43:
                pd_data+=row
            elif i==43:
                pd_valid_lens+=row
                data1_len=len(pd_labels)
                # print(len(pd_labels))
            elif i==44:
                patients+=row
            elif i==45:
                sessions+=row
            elif i >=46 and i<50:
                pd_labels+=row
            elif i>=50 and i<54:
                pd_data+=row
            elif i==54:
                pd_valid_lens+=[int(i1)+data1_len for i1 in row]
            elif i==55:
                patients+=row
            elif i==56:
                sessions+=row
    pd_valid_lens.append(len(pd_labels))
    for i in range(len(pd_valid_lens)):
        pd_valid_lens[i]=int(pd_valid_lens[i])
    for i in range(len(pd_data)):
        pd_data[i] = float(pd_data[i])
    print('patients',len(patients))
    print('unique_patients',len(set(patients)))
    print('sessions',len(sessions))
    print('unique_sessions',len(set(sessions)))
    true_num=0
    for i in range(len(pd_labels)):
        if pd_labels[i]=='True':
            pd_labels[i]=1
            true_num+=1
        else:
            pd_labels[i] = 0
    print(len(pd_labels))
    print(sum(pd_labels))
    print('true_ratio:',float(true_num/len(pd_labels)))
    # results= pd.read_csv(filename)
    # print(results)
    # print(results.loc[2,'16000'])
    labels=[]
    valid_lens=[]
    data=[]
    # pd_valid_lens=results.loc[2,:].tolist()[:total+1]
    # for i in range(len(pd_valid_lens)):
    #     pd_valid_lens[i]=int(pd_valid_lens[i])
    # pd_data=results.loc[1,:].tolist()
    # for i in range(len(pd_data)):
    #     pd_data[i]=float(pd_data[i])
    # pd_labels=results.loc[0,:].tolist()
    # for i in range(len(pd_labels)):
    #     if pd_labels[i]=='FALSE':
    #         pd_labels[i]=0
    #     else:
    #         pd_labels[i] = 1

    def creat_his(raw,length=10):
        his=np.zeros(length)
        for i in raw:
            his[int(i// (1 / length + 0.001))] += 1
        # print(his)
        his=his/sum(his)
        # print(his)
        return his
    def creat_his_by_criterion(criterion):
        for i in set(criterion):
            indexes=findall(criterion,i)
            # print(indexes)
            data_pa=[]
            valid_len=0
            for index in indexes:
                data_pa+=pd_data[pd_valid_lens[index]:pd_valid_lens[index+1]]
                valid_len+=pd_valid_lens[index+1]-pd_valid_lens[index]
            # print(len(data_pa)==valid_len)
            data.append(creat_his(data_pa,length))
            valid_lens.append(valid_len)
            labels.append(pd_labels[pd_valid_lens[indexes[0]]])


    sessions_patients=[]
    for i,patient in enumerate(patients):
        sessions_patients.append(str(patient)+str(sessions[i]))

    # print(pd_valid_lens[-10:])
    if use_session_or_patients=='patients':
        creat_his_by_criterion(patients)
    elif use_session_or_patients=='sessions':
        creat_his_by_criterion(sessions_patients)
    else:

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

    # for i,d in enumerate(data):
    #     if len(d)!=30:
    #         print(d)
    #         print(valid_lens[i])
    # print(labels)
    # print(valid_lens)
    # print(data)
    print(len(labels))
    print(len(valid_lens))
    print(len(data))
    # print('min',min(valid_lens))
    class MyDataset(Dataset):
        def __init__(self,data,labels,valid_lens):
            self.valid_lens=valid_lens
            self.labels=labels
            self.data=torch.tensor(data,dtype=torch.float)
        def __len__(self):
            return len(self.data)
        def __getitem__(self, index):
            return self.data[index],self.labels[index],self.valid_lens[index]

    dataset = MyDataset(data,labels,valid_lens)
    ave_test_acc=0
    ave_ori_acc=0
    ave_argmax_acc=0
    ave_mean_acc=0
    for n_time in range(n_repetition):
        idx_train, idx_test = train_test_split(np.arange(len(valid_lens)),
                                                     random_state=n_time,
                                                     train_size=0.8,
                                                     shuffle=True)
        idx_train, idx_valid = train_test_split(idx_train,
                                                     random_state=n_time,
                                                     train_size=0.75,
                                                     shuffle=True)
        test_set = torch.utils.data.Subset(dataset,idx_test)
        valid_set=torch.utils.data.Subset(dataset,idx_valid)
        train_set = torch.utils.data.Subset(dataset, idx_train)
        #将数据集导入DataLoader，进行shuffle以及选取batch_size
        data_loader = DataLoader(train_set,batch_size=1,shuffle=True,num_workers=0)
        data_loader_valid=DataLoader(valid_set,batch_size=1,shuffle=True,num_workers=0)


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
        if use_session_or_patients:
            model=histogram_model(length=length,use_hybrid=False)
        elif use_his:
            model=histogram_model(length=length,use_hybrid=use_hybrid)
        else:
            model=decision_model(adap_pool=adap_pool)
        print(model)
        # datum1,label1,valid_len1=next(iter(data_loader))
        # print(model(datum1))
        cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
        device = 'cuda' if cuda else 'cpu'
        model.to(device)
        lr=0.001
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        T_max=20
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)

        loss = torch.nn.NLLLoss()
        model.train()
        num_epochs=20

        min_loss_val = 1000000
        best_model=None
        for epoch in range(num_epochs):
            all=0
            right=0
            total_loss=0
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
                total_loss+=l
                l.backward()  # Make the loss scalar for `backward`
                optimizer.step()
            if (epoch+1)%2==0:
                print('epoch:',epoch+1)
                print('loss:',total_loss)
                print('acc',right/all)

            all=0
            right=0
            total_loss=0
            with torch.no_grad():
                for batch in data_loader_valid:
                    X, Y, valid_len = [x.to(device) for x in batch]
                    Y_hat = model(X, valid_len)
                    hat_label = torch.argmax(Y_hat, -1)
                    res = sum(hat_label == Y)
                    all += len(Y)
                    right += res
                    # print('hat', Y_hat.shape)
                    l = loss(Y_hat, Y)
                    total_loss+=l
            if total_loss<min_loss_val:
                best_model=copy.deepcopy(model)
                min_loss_val=total_loss
            if (epoch+1)%2==0:
                print('valid_loss:',total_loss)
                print('valid_acc',right/all)

        model=best_model
        print('min_loss_val',min_loss_val)
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
        writer.writerow(['use_hybrid','n_repetition','length','use_his','adap_pool','ave_test_acc','ave_ori_acc','ave_argmax_acc','ave_mean_acc','use_session','total'])
        writer.writerow([use_hybrid,n_repetition,length,use_his,adap_pool,ave_test_acc,ave_ori_acc,ave_argmax_acc,ave_mean_acc,use_session_or_patients,total])













