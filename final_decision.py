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
import time
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from util import *
from itertools import product

N_REPETITION=[5]
LENGTH=[10]#2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
USE_HIS=[True]
ADAP_POOL=[False]
USE_HYBRID=[False]
TOTAL=[2986]#2986,21629
NUM_EPOCHS=[60]#20,40,60,80
USE_SESSION_OR_PATIENTS=['patients','sessions',None]#'patients','sessions',None
LR=[0.01]#0.001,0.002,0.005,0.01  tuab:(0.01,64)  tueg(0.005,256)
FIX_TESTSET=[False,True]#True,False
BATCH_SIZE=[64]#8,16,32,64,128,256,512,1024
BLOCK=[0,1,2,3,4,5,6,7]#0,1,2,3,4,5,6,7
for (n_repetition,length,use_his,adap_pool,use_hybrid,total,use_session_or_patients,batch_size,num_epochs,lr,fix_testset,block) in product(N_REPETITION,\
    LENGTH,USE_HIS,ADAP_POOL,USE_HYBRID,TOTAL,USE_SESSION_OR_PATIENTS,BATCH_SIZE,NUM_EPOCHS,LR,FIX_TESTSET,BLOCK):
    print('batch_size',batch_size)
    print('num_epochs',num_epochs)
    print('lr',lr)
    tuab=True
    tueg=False
    deep4=False
    tcn_1=True
    if tuab:
        filename = './training_detail_tuab.csv'

    if tueg:
        filename = './training_detail_tueg.csv'
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
            if tueg:
                if deep4:
                    if i >=0 and i<16:
                        pd_labels+=row
                    elif i>=16 and i<32:
                        pd_data+=row
                    elif i==32:
                        pd_valid_lens+=row
                        data1_len=len(pd_labels)
                        print(len(pd_labels))
                    elif i==33:
                        patients+=row
                    elif i==34:
                        sessions+=row
                    if tcn_1:
                        if i >= 35 and i < 51:
                            pd_labels += row
                        elif i >= 51 and i < 67:
                            pd_data += row
                        elif i == 67:
                            pd_valid_lens+=[int(i1)+data1_len for i1 in row]
                        elif i == 68:
                            patients += row
                        elif i == 69:
                            sessions += row
                if tcn_1:
                    if i >= 35 and i < 51:
                        pd_labels += row
                    elif i >= 51 and i < 67:
                        pd_data += row
                    elif i == 67:
                        pd_valid_lens += row
                        data1_len = len(pd_labels)
                        print(len(pd_labels))
                    elif i == 68:
                        patients += row
                    elif i == 69:
                        sessions += row
            if tuab:
                if i >= 35+block*11 and i < 39+block*11:
                    pd_labels += row
                elif i >= 39+block*11 and i < 43+block*11:
                    pd_data += row
                elif i == 43+block*11:
                    pd_valid_lens += row
                elif i == 44+block*11:
                    patients += row
                elif i == 45+block*11:
                    sessions += row


    pd_valid_lens.append(len(pd_labels))


    for i in range(len(pd_valid_lens)):
        pd_valid_lens[i]=int(pd_valid_lens[i])
    for i in range(len(pd_data)):
        pd_data[i] = float(pd_data[i])

    print('recordings',len(patients))
    print('unique_patients',len(set(patients)))
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
    print('unique_sessions',len(set(sessions_patients)))
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
    # labels=labels[:-4000]
    # valid_lens=valid_lens[:-4000]
    # data=data[:-4000]

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
                                                     shuffle=not fix_testset)
        # print(idx_train)
        idx_train, idx_valid = train_test_split(idx_train,
                                                     random_state=n_time,
                                                     train_size=0.75,
                                                     shuffle=True)
        test_set = torch.utils.data.Subset(dataset,idx_test)
        valid_set=torch.utils.data.Subset(dataset,idx_valid)
        train_set = torch.utils.data.Subset(dataset, idx_train)

        data_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=0)
        data_loader_valid=DataLoader(valid_set,batch_size=batch_size,shuffle=True,num_workers=0)


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
                    # self.classifer=nn.Sequential(
                    #     nn.Linear(20,20),
                    #     nn.ReLU(),
                    #     nn.Linear(20,2)
                    # )
                    self.classifer=nn.Linear(20,2)
                self.log_softmax = nn.LogSoftmax(dim=1)
                init.normal_(self.classifer.weight, 0, 0.01)
                # for layer in self.classifer:
                #     if isinstance(layer, nn.Linear):
                #         layer.weight.data.normal_()
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

        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=0.01)
        T_max=num_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)

        loss = torch.nn.NLLLoss()
        model.train()


        min_loss_val = 1000000
        best_model=None
        iters = len(data_loader)
        for epoch in range(num_epochs):
            all=0
            right=0
            total_loss=0
            for i,batch in enumerate(data_loader):
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
                scheduler.step(epoch + i / iters)
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
        data_loader = DataLoader(test_set,batch_size=16,shuffle=False,num_workers=0)
        for batch in data_loader:
            X, Y, valid_len = [x.to(device) for x in batch]
            compare_res=[]
            X1=X>0.5
            for i,j,h in zip(X1,Y,valid_len):
                mid=i==j
                mid1=mid[:h]
                compare_res.append(sum((i==j)[:h]))
            # compare_res=torch.sum((X > 0.5)==Y,dim=-1)
            right_ori+=sum(compare_res)
            all_ori += sum(valid_len)
            for i,j in zip(compare_res,valid_len):
                if i>j/2:
                    right_argmax+=1
                all_argmax+=1
            mean_res=torch.mean(X,dim=-1)
            for i,j,h in zip(mean_res,Y,valid_len):
                if (i>(0.5*h/20))==j:
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
        writer.writerow(['use_hybrid','n_repetition','length','use_his','adap_pool','ave_test_acc',\
                         'ave_ori_acc','ave_argmax_acc','ave_mean_acc','use_session','total','epochs',\
                         'time','batch_size','num_epochs','lr','fix_testset','deep4','tcn_1','tuab','tueg',\
                         'block'])
        writer.writerow([use_hybrid,n_repetition,length,use_his,adap_pool,ave_test_acc,ave_ori_acc,\
                         ave_argmax_acc,ave_mean_acc,use_session_or_patients,total,num_epochs,\
                         time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time())),batch_size,\
                         num_epochs,lr,fix_testset,deep4,tcn_1,tuab,tueg,block])













