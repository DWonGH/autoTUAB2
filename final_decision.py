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
from collections import OrderedDict
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from util import *
from itertools import product

N_REPETITION=[5]
LENGTH=[10]#2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
USE_HIS=[False]
ADAP_POOL=[False]
USE_HYBRID=[False]
TOTAL=[2910]#2986,2910,26504
NUM_EPOCHS=[60]#20,40,60,80
USE_SESSION_OR_PATIENTS=[None]#'patients','sessions',None
LR=[0.005]#0.001,0.002,0.005,0.01  tuab:(0.01,64)  tueg(0.005,256)
FIX_TESTSET=[True]#True,False
BATCH_SIZE=[64]#8,16,32,64,128,256,512,1024
BLOCK=[0,1,2,3,4]#0,1,2,3,4
HIDDEN_LAYERS=[0]
HIDDEN_LENGTH=[5]
MUL_THRESH=[0.5]#0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
# XGBOOST=[True]
xgboost=False
save_each_experiment=False

# xgboost=False
START_ROW_GAP=[[546,4,4,20],]  #[266,4,3],[412,2,3],[381,1,3],[584,4,4],[544,2,4],[484,1,4]    [1,4,4],[91,1,4],[121,2,4],[231,4,4],[191,2,4],[161,1,4]   [546,4,4],[694,2,4],[734,1,4],[824,1,4],[764,1,4] 400,600  [546,4,4,20],[694,2,4,6],[734,1,4,4],[824,1,4,3],[764,1,4,2]
#[1040,25,4]   [1738,25,4]
# [1310,4,4],[2164,1,4],[2194,1,4]  tcn
# [1650,4,4],[1698,2,4 ] ,[91,1,4]   leaf_efficientnet
# [31,4,4]   leaf_efficientnet 60
# [1530,4,4],    vit small
#[1590,4,4],[2008,1,4]   vit tiny
# [2038,4,4]  passt
# [1,1,4,2]  deep4 pre 600
# [403,4,4]
def remove_all(list1,num):
    while(num in list1):
        list1.remove(num)
    return list1


with open('./decision_result.csv', 'a') as f:
    writer = csv.writer(f, delimiter=',', lineterminator='\n', )
    # writer.writerow([time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))])
    if not xgboost:
        writer.writerow(['use_hybrid', 'n_repetition', 'length', 'use_his', 'adap_pool', 'ave_test_acc', \
                         'ave_ori_acc', 'ave_argmax_acc', 'ave_mean_acc', 'use_session', 'total', 'epochs', \
                         'time', 'batch_size', 'num_epochs', 'lr', 'fix_testset', 'start', 'rows', \
                         'block', 'hidden_layers', 'hidden_length', 'ave_test_sen', 'ave_test_spe', 'ave_ori_sen',
                         'ave_ori_spe', 'ave_mean_sen', 'ave_mean_spe', 'con_matrix', 'con_matrix_mean', 'con_matrix_no','ave_mul_acc1','ave_mul_sen','ave_mul_spe'])
    else:
        writer.writerow(['n_repetition', 'accuracy', \
                     # 'train_best_score',\
                     'Best_parameters_set', \
                     'time', 'start', 'rows', \
                     'block', 'use_his', 'use_hybrid', 'use_session_or_patients'])
    for (n_repetition,length,use_his,adap_pool,use_hybrid,total,use_session_or_patients,batch_size,num_epochs,lr,fix_testset,block,hidden_layers,hidden_length,start_row_gap,mul_thresh) in product(N_REPETITION,\
        LENGTH,USE_HIS,ADAP_POOL,USE_HYBRID,TOTAL,USE_SESSION_OR_PATIENTS,BATCH_SIZE,NUM_EPOCHS,LR,FIX_TESTSET,BLOCK,HIDDEN_LAYERS,HIDDEN_LENGTH,START_ROW_GAP,MUL_THRESH):
        print('batch_size',batch_size)
        print('num_epochs',num_epochs)
        print('lr',lr)
        start=start_row_gap[0] #deep4: 266,4  412,2  381,1    tcn: 584,4   544,2    484,1
        rows=start_row_gap[1]
        if rows==4:
            total=2986
        filename = './training_detail_2023_2_2.csv'
        # filename = './training_detail.csv'

        print('start_row_gap',start_row_gap)
        # print(block)
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
        rows_total=rows*2+start_row_gap[2]
        with open(filename, newline='') as csvfile:
            results = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(results):
                # print(i)
                # print(row)

                if i >= start+block*rows_total and i < start+rows+block*rows_total:
                    pd_labels += remove_all(row,'')
                elif i >= start+rows+block*rows_total and i < start+rows*2+block*rows_total:
                    pd_data += remove_all(row,'')
                elif i == start+rows*2+block*rows_total:
                    pd_valid_lens += remove_all(row,'')
                elif i == start+rows*2+1+block*rows_total:
                    patients += remove_all(row,'')
                elif i == start+rows*2+2+block*rows_total:
                    sessions += remove_all(row,'')


        pd_valid_lens.append(len(pd_labels))
        # print('pd_labels',pd_labels[:5])
        # print('pd_data',pd_data[:5])

        for i in range(len(pd_valid_lens)):
            # print(pd_valid_lens[i])
            pd_valid_lens[i]=int(pd_valid_lens[i])

        for i in range(len(pd_data)):
            pd_data[i] = float(pd_data[i])

        print('recordings',len(patients))
        print('unique_patients',len(set(patients)))
        # print('patients',patients[:5])
        print('unique_sessions',len(set(sessions)))
        # print('sessions',sessions[:5])
        true_num=0
        for i in range(len(pd_labels)):
            # print(pd_labels[i])
            if pd_labels[i]=='True' or pd_labels[i]=='TRUE':
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
                # if i>1:
                #     print('exception',i)
                # else:
                his[int(i// (1 / length + 0.001))] += 1
            # print(his)
            his=his/sum(his)
            # print(his)
            return his
        def creat_his_by_criterion(criterion):
            # for i in set(criterion):
            for i in OrderedDict.fromkeys(criterion).keys():
                indexes=findall(criterion,i)
                # print(indexes)
                # print(i)
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
                test_label=pd_labels[pd_valid_lens[indexes[0]]]
                for index in indexes:
                    if test_label != pd_labels[pd_valid_lens[index]]:
                        print(criterion[index])
                        print(criterion[indexes[0]])
                        print(pd_labels[pd_valid_lens[index]-10:pd_valid_lens[index]],pd_labels[pd_valid_lens[index]:pd_valid_lens[index]+10])

                        print(pd_labels[pd_valid_lens[indexes[0]]-10:pd_valid_lens[indexes[0]]],pd_labels[pd_valid_lens[indexes[0]]:pd_valid_lens[indexes[0]]+10])

        max_valid_len=start_row_gap[3]
        print('max_valid_len',max_valid_len)
        sessions_patients=[]
        for i,patient in enumerate(patients):
            sessions_patients.append(str(patient)+str(sessions[i]))
        print('unique_sessions',len(set(sessions_patients)))
        print('sessions',len(sessions_patients))

        # print('unique_sessions',sessions_patients[:5])
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
                        data.append(np.concatenate([creat_his(pd_data[pd_valid_lens[i]:pd_valid_lens[i+1]],length=length),pd_data[pd_valid_lens[i]:pd_valid_lens[i+1]]+[0]*(max_valid_len-valid_lens[-1])],axis=0)  )
                    else:
                        data.append(creat_his(pd_data[pd_valid_lens[i]:pd_valid_lens[i+1]],length=length))
                else:
                    data.append(pd_data[pd_valid_lens[i]:pd_valid_lens[i+1]]+[0]*(max_valid_len-valid_lens[-1]))


        print('labels',type(labels),np.array(labels).shape)
        print('data',type(data),np.array(data).shape)
        if rows == 4:
            train_ratio = 0.9079
        elif rows==25:
            train_ratio=0.9
        else:
            train_ratio = 0.9072

        if xgboost:
            import xgboost as xgb
            from sklearn.metrics import accuracy_score
            import matplotlib.pyplot as plt


            train_len=int(len(labels)*train_ratio)
            # train_set= xgb.DMatrix(np.array(data[:train_len]), label=np.array(labels[:train_len]))
            # test_set= xgb.DMatrix(np.array(data[train_len:]))
            # test_label=np.array(labels[train_len:])
            # # # for i in range(n_repetition):
            # params = {
            #             'booster': 'gbtree',
            #             'objective': 'multi:softmax',
            #             'num_class': 2,
            #             'gamma': 0.1,
            #             'max_depth': 6,
            #             'lambda': 2,
            #             'subsample': 0.7,
            #             'colsample_bytree': 0.75,
            #             'min_child_weight': 3,
            #             'silent': 0,
            #             'eta': 0.1,
            #             'seed': 1,
            #             'nthread': 4,
            #         }
            # plst = list(params.items())
            # num_rounds = 500
            # model = xgb.train(plst, train_set, num_rounds)
            # y_pred = model.predict(test_set)
            #
            #
            # accuracy = accuracy_score(test_label,y_pred)
            # print("accuarcy: %.2f%%" % (accuracy*100.0))
            #
            #     xgb.plot_importance(model)
            # plt.show()
            #
            from sklearn.model_selection import GridSearchCV
            train_x=np.array(data[:train_len])
            train_y=np.array(labels[:train_len])
            test_x=np.array(data[train_len:])
            test_y=np.array(labels[train_len:])
            #
            #
            # parameters = {
            #               'max_depth': [5, 10, 15, 20, 25],
            #               'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
            #               'n_estimators': [500, 1000, 2000, 3000, 5000],
            #               'min_child_weight': [0, 2, 5, 10, 20],
            #               # 'max_delta_step': [0, 0.2, 0.6, 1, 2],
            #               # 'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
            #               # 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
            #               # 'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
            #               # 'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
            #               # 'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]
            #
            # }
            parameters = {
                          'max_depth': [5, 10, 15, 20, 25],


            }

            xlf = xgb.XGBClassifier(max_depth=10,
                        learning_rate=0.01,
                        n_estimators=2000,
                        silent=True,
                        objective='multi:softmax',
                        num_class=2 ,
                        nthread=-1,
                        gamma=0,
                        min_child_weight=1,
                        max_delta_step=0,
                        subsample=0.85,
                        colsample_bytree=0.7,
                        colsample_bylevel=1,
                        reg_alpha=0,
                        reg_lambda=1,
                        scale_pos_weight=1,
                        seed=0,
                        )

            gs = GridSearchCV(xlf, param_grid=parameters, scoring='accuracy', cv=3)
            gs.fit(train_x, train_y)


            print("Best score: %0.3f" % gs.best_score_)
            print("Best parameters set: %s" % gs.best_params_ )
            y_pred=gs.predict(test_x)
            accuracy = accuracy_score(test_y,y_pred)
            print("accuarcy: %.2f%%" % (accuracy*100.0))
            # with open('./decision_result.csv','a') as f:
            #     writer=csv.writer(f, delimiter=',',lineterminator='\n',)
            #     # writer.writerow([time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))])
            #

            writer.writerow([n_repetition,accuracy,\
                             # gs.best_score_,\
                             gs.best_params_,\
                             time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time())),\
                             start,rows,block,use_his,use_hybrid,use_session_or_patients])
        else:
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
                def __getitem__(self, index):
                    return self.data[index],self.labels[index],self.valid_lens[index]

            dataset = MyDataset(data,labels,valid_lens)
            ave_test_acc=0
            ave_ori_acc=0
            ave_argmax_acc=0
            ave_mean_acc=0
            ave_test_mean_acc=0
            ave_test_abnormal_acc=0
            split_way=None
            con_matrix=[0,0,0,0]
            con_matrix_no=[0,0,0,0]
            con_matrix_mean=[0,0,0,0]
            con_matrix_multiply=[0,0,0,0]
            for n_time in range(n_repetition):
                print('n_time',n_time)
                if not split_way:
                    idx_train, idx_test = train_test_split(np.arange(len(valid_lens)),
                                                           random_state=n_time,
                                                           train_size=train_ratio,
                                                           shuffle=not fix_testset)

                    # if fix_testset:
                    #     if rows==4:
                    #         idx_train=range(2986)[:2709]
                    #         idx_test=range(2986)[2709:]
                    #     else:
                    #         idx_train=range(2910)[:2642]
                    #         idx_test=range(2910)[2642:]
                    # else:
                    #     idx_train, idx_test = train_test_split(np.arange(len(valid_lens)),
                    #                                              random_state=n_time,
                    #                                              train_size=0.9,
                    #                                              shuffle=not fix_testset)
                    # print(idx_train)
                    idx_train, idx_valid = train_test_split(idx_train,
                                                            random_state=n_time,
                                                            train_size=0.85,
                                                            shuffle=True)

                else:
                    if split_way=='session':
                        criterion=sessions_patients
                    unique_criterion=list(OrderedDict.fromkeys(criterion).keys())
                    idx_train_cri, idx_test_cri = train_test_split(np.arange(len(unique_criterion )),
                                                           random_state=n_time,
                                                           train_size=train_ratio,
                                                           shuffle=not fix_testset)

                    idx_train_cri, idx_valid_cri = train_test_split(idx_train_cri,
                                                            random_state=n_time,
                                                            train_size=0.85,
                                                            shuffle=True)
                    idx_train=[]
                    idx_valid=[]
                    idx_test=[]
                    for idx in idx_train_cri:
                        idx_train+=findall(criterion, unique_criterion[idx])
                    for idx in idx_valid_cri:
                        idx_valid+=findall(criterion, unique_criterion[idx])
                    for idx in idx_test_cri:
                        idx_test+=findall(criterion, unique_criterion[idx])
                print('idx_train',len(idx_train))
                print('idx_valid',len(idx_valid))
                print('idx_test',len(idx_test))


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
                            self.classifer=nn.Linear(max_valid_len,2)
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
                    def __init__(self,length=10,use_hybrid=False,hidden_layers=hidden_layers,hidden_length=hidden_length):
                        super().__init__()
                        if use_hybrid:
                            self.length=length+max_valid_len
                        else:
                            self.length=length
                        self.hidden_length=hidden_length
                        self.hidden_layers=hidden_layers
                        if self.hidden_layers>0:
                            self.hidden=nn.Sequential()
                            self.hidden.add_module("hidden{:d}".format(0),nn.Linear(self.length,self.hidden_length))
                            self.hidden.add_module("activation{:d}".format(0),nn.ReLU())
                            self.hidden_layers=self.hidden_layers-1
                            for i in range(self.hidden_layers):
                                self.hidden.add_module("hidden{:d}".format(i+1),nn.Linear(self.hidden_length,self.hidden_length))
                                self.hidden.add_module("activation{:d}".format(i+1),nn.ReLU())

                            for subhidden in self.hidden:
                                if hasattr(subhidden,'weight'):
                                    init.normal_(subhidden.weight, 0, 0.01)
                            self.hidden_layers=self.hidden_layers+1
                        if self.hidden_layers>0:
                            self.classifer=nn.Linear(self.hidden_length,2)
                        else:
                            self.classifer=nn.Linear(self.length,2)
                        self.log_softmax = nn.LogSoftmax(dim=1)


                        init.normal_(self.classifer.weight, 0, 0.01)

                    def forward(self,x,valid_len):
                        if self.hidden_layers>0:
                            x=self.hidden(x)
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
                    if (epoch+1)%5==0:
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
                    if (epoch+1)%5==0:
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
                hat_labels_list=[]
                hat_prob_list=[]
                test_labels=labels[-2651:]
                test_sessions=sessions_patients[-2651:]
                for batch in data_loader:
                    X, Y, valid_len = [x.to(device) for x in batch]
                    if not use_his:
                        compare_res=[]
                        X1=X>0.5
                        for i,j,h in zip(X1,Y,valid_len):
                            # print('ori_res',sum((i==j)[:h]))
                            # print('valid_len',h)
                            compare_res.append(sum((i==j)[:h]))
                            for h1 in range(h):
                                if j:
                                    if i[h1]:
                                        con_matrix_no[0]+=1
                                    else:
                                        con_matrix_no[1]+=1
                                else:
                                    if i[h1]:
                                        con_matrix_no[2]+=1
                                    else:
                                        con_matrix_no[3]+=1

                        # compare_res=torch.sum((X > 0.5)==Y,dim=-1)
                        right_ori+=sum(compare_res)
                        all_ori += sum(valid_len)
                        for i,j in zip(compare_res,valid_len):
                            if i>j/2:
                                right_argmax+=1
                            all_argmax+=1
                        mean_res=torch.mean(X,dim=-1)
                        for i,j,h in zip(mean_res,Y,valid_len):
                            if (i>(0.5*h/max_valid_len))==j:
                                right_mean+=1
                            all_mean+=1
                            if j:
                                if (i>(0.5*h/max_valid_len)):
                                    con_matrix_mean[0]+=1
                                else:
                                    con_matrix_mean[1]+=1
                            else:
                                if (i>(0.5*h/max_valid_len)):
                                    con_matrix_mean[2]+=1
                                else:
                                    con_matrix_mean[3]+=1
                        # print('X',X.shape)
                        multiply_res = []
                        for i in range(len(valid_len)):
                            valid_len_1 = valid_len[i]
                            mutiply = 1
                            for j in range(valid_len_1):
                                mutiply *= 1 - X[i][j]
                            mutiply = mutiply**(1 / valid_len_1)
                            print('mutiply',mutiply)
                            multiply_res.append(mutiply < mul_thresh)
                        # print('multiply_res',multiply_res)
                        for i,j in zip(multiply_res,Y):
                            if j:
                                if i:
                                    con_matrix_multiply[0]+=1
                                else:
                                    con_matrix_multiply[1]+=1
                            else:
                                if i:
                                    con_matrix_multiply[2]+=1
                                else:
                                    con_matrix_multiply[3]+=1

                    Y_hat = model(X, valid_len)
                    # print('Y_hat',torch.exp(Y_hat))
                    hat_prob=torch.exp(Y_hat)[:,1]
                    hat_prob_list+=hat_prob
                    hat_label = torch.argmax(Y_hat, -1)

                    hat_labels_list+=hat_label
                    res = sum(hat_label == Y)

                    all += len(Y)
                    right += res
                    for i in range(len(hat_label)):
                        if Y[i]:
                            if hat_label[i]:
                                con_matrix[0]+=1
                            else:
                                con_matrix[1]+=1
                        else:
                            if hat_label[i]:
                                con_matrix[2]+=1
                            else:
                                con_matrix[3]+=1





                # if not use_his:
                #     # print('hat_labels_list',len(hat_labels_list))
                #     # print('test_sessions',len(set(test_sessions)))
                #     # print('hat_prob_list',len(hat_prob_list))
                #     hat_labels_list_1 = []
                #     hat_labels_list_2 = []
                #     test_labels_1 = []
                #     for i in OrderedDict.fromkeys(test_sessions).keys():
                #         indexes = findall(test_sessions, i)
                #         # print(indexes)
                #         # print(i)
                #         # print(indexes)
                #         test_label=test_labels[indexes[0]]
                #         for index in indexes:
                #             if test_label!=test_labels[index]:
                #                 print(test_sessions[index])
                #                 print(test_sessions[indexes[0]])
                #                 print(test_labels[index])
                #                 print(test_label)
                #
                #         test_labels_1.append(test_label)
                #         # hat_label=0
                #         # for index in indexes:
                #         #     if hat_labels_list[index]:
                #         #         hat_label=1
                #         #         break
                #         prob=0
                #         # print('hat_prob_list',len(hat_prob_list))
                #         hat_label_2=0
                #         for index in indexes:
                #             if hat_labels_list[index]:
                #                 hat_label_2=1
                #             # print(index)
                #             prob+=hat_prob_list[index]
                #             print('prob',prob)
                #         # hat_label_2=hat_prob_list[indexes[0]]>0.5
                #         if prob/len(indexes) >0.5:
                #             hat_label=1
                #         else:
                #             hat_label=0
                #         print(hat_label)
                #         print(hat_label_2)
                #
                #         hat_labels_list_2.append(hat_label_2==test_labels[indexes[0]])
                #         hat_labels_list_1.append(hat_label==test_labels[indexes[0]])
                    # print('test_labels_1',len(test_labels_1))
                    # print('hat_labels_list_1',len(hat_labels_list_1))
                    # print('hat_labels_list_2',len(hat_labels_list_2))
                    # print('session_test_acc_mean',sum(hat_labels_list_1)/len(test_labels_1))
                    # print('session_test_acc_abnornal',sum(hat_labels_list_2)/len(test_labels_1))



                if save_each_experiment:
                    test_sen = float(con_matrix[0] / (con_matrix[0] + con_matrix[1]))
                    test_spe = float(con_matrix[0] / (con_matrix[0] + con_matrix[2]))
                    test_acc1 = float(
                        (con_matrix[0] + con_matrix[3]) / (con_matrix[0] + con_matrix[3] + con_matrix[2] + con_matrix[1]))
                    print('test_sen', test_sen)
                    print('test_spe', test_spe)
                    print('test_acc1', test_acc1)
                    ori_acc1,ori_sen, ori_spe, mean_acc1,mean_sen, mean_spe = 0, 0, 0, 0,0,0
                    if not use_his:
                        ori_sen = float(con_matrix_no[0] / (con_matrix_no[0] + con_matrix_no[1]))
                        ori_spe = float(con_matrix_no[0] / (con_matrix_no[0] + con_matrix_no[2]))
                        ori_acc1 = float(
                            (con_matrix_no[0] + con_matrix_no[3]) / (
                                        con_matrix_no[0] + con_matrix_no[3] + con_matrix_no[2] + con_matrix_no[1]))
                        print('ave_ori_sen', ori_sen)
                        print('ave_ori_spe', ori_spe)
                        print('ave_ori_acc1', ori_acc1)
                        mean_sen = float(con_matrix_mean[0] / (con_matrix_mean[0] + con_matrix_mean[1]))
                        mean_spe = float(con_matrix_mean[0] / (con_matrix_mean[0] + con_matrix_mean[2]))
                        mean_acc1 = float((con_matrix_mean[0] + con_matrix_mean[3]) / (
                                    con_matrix_mean[0] + con_matrix_mean[3] + con_matrix_mean[2] + con_matrix_mean[1]))
                        print('mean_sen', mean_sen)
                        print('ave_mean_spe', mean_spe)
                        print('mean_acc1', mean_acc1)
                    writer.writerow([use_hybrid, n_time, length, use_his, adap_pool, test_acc1, ori_acc1,''\
                                     , mean_acc1, use_session_or_patients, total, num_epochs, \
                                     time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time())), batch_size, \
                                     num_epochs, lr, fix_testset, start, rows, block, hidden_layers, hidden_length,
                                     test_sen, test_spe, ori_sen, ori_spe, mean_sen, mean_spe,
                                     con_matrix, con_matrix_mean, con_matrix_no])
                    con_matrix = [0, 0, 0, 0]
                    con_matrix_no = [0, 0, 0, 0]
                    con_matrix_mean = [0, 0, 0, 0]
                else:
                    print('test_acc', right / all)
                    ave_test_acc += right / all
                    if not use_his:
                        print('ori_acc', right_ori / all_ori)
                        print('argmax_acc', right_argmax / all_argmax)
                        print('mean_acc', right_mean / all_mean)

                        ave_ori_acc += right_ori / all_ori
                        ave_argmax_acc += right_argmax / all_argmax
                        ave_mean_acc += right_mean / all_mean
                # if not use_his:
                #     ave_test_mean_acc+=sum(hat_labels_list_1)/len(test_labels_1)
                #     ave_test_abnormal_acc+=sum(hat_labels_list_2)/len(test_labels_1)
            if not save_each_experiment:
                ave_test_acc=float(ave_test_acc/n_repetition)
                print('ave_test_acc',ave_test_acc)
                if not use_his:
                    ave_ori_acc=float(ave_ori_acc/n_repetition)
                    ave_argmax_acc=float(ave_argmax_acc/n_repetition)
                    ave_mean_acc=float(ave_mean_acc/n_repetition)

                    print('ave_ori_acc',ave_ori_acc)
                    print('ave_argmax_acc',ave_argmax_acc)
                    print('ave_mean_acc',ave_mean_acc)
                ave_test_sen=float(con_matrix[0]/(con_matrix[0]+con_matrix[1]))
                ave_test_spe=float(con_matrix[0]/(con_matrix[0]+con_matrix[2]))
                ave_test_acc1=float((con_matrix[0]+con_matrix[3])/(con_matrix[0]+con_matrix[3]+con_matrix[2]+con_matrix[1]))
                print('ave_test_sen',ave_test_sen)
                print('ave_test_spe',ave_test_spe)
                print('ave_test_acc1',ave_test_acc1)
                ave_ori_sen, ave_ori_spe, ave_mean_sen, ave_mean_spe=0,0,0,0
                if not use_his:
                    ave_ori_sen = float(con_matrix_no[0] / (con_matrix_no[0] + con_matrix_no[1]))
                    ave_ori_spe = float(con_matrix_no[0] / (con_matrix_no[0] + con_matrix_no[2]))
                    ave_ori_acc1 = float(
                        (con_matrix_no[0] + con_matrix_no[3]) / (con_matrix_no[0] + con_matrix_no[3] + con_matrix_no[2] + con_matrix_no[1]))
                    print('ave_ori_sen', ave_ori_sen)
                    print('ave_ori_spe', ave_ori_spe)
                    print('ave_ori_acc1', ave_ori_acc1)
                    # print('con_matrix_mean',con_matrix_mean)
                    ave_mean_sen=float(con_matrix_mean[0]/(con_matrix_mean[0]+con_matrix_mean[1]))
                    ave_mean_spe=float(con_matrix_mean[0]/(con_matrix_mean[0]+con_matrix_mean[2]))
                    ave_mean_acc1=float((con_matrix_mean[0]+con_matrix_mean[3])/(con_matrix_mean[0]+con_matrix_mean[3]+con_matrix_mean[2]+con_matrix_mean[1]))
                    print('ave_mean_sen',ave_mean_sen)
                    print('ave_mean_spe',ave_mean_spe)
                    print('ave_mean_acc1',ave_mean_acc1)
                    ave_mul_sen=float(con_matrix_multiply[0]/(con_matrix_multiply[0]+con_matrix_multiply[1]))
                    ave_mul_spe=float(con_matrix_multiply[0]/(con_matrix_multiply[0]+con_matrix_multiply[2]))
                    ave_mul_acc1=float((con_matrix_multiply[0]+con_matrix_multiply[3])/(con_matrix_multiply[0]+con_matrix_multiply[3]+con_matrix_multiply[2]+con_matrix_multiply[1]))
                    print('ave_mul_sen',ave_mul_sen)
                    print('ave_mul_spe',ave_mul_spe)
                    print('ave_mul_acc1',ave_mul_acc1)
                # if not use_his:
                #     ave_test_mean_acc=float(ave_test_mean_acc/n_repetition)
                #     ave_test_abnormal_acc=float(ave_test_abnormal_acc/n_repetition)
                #     print('ave_test_mean_acc',ave_test_mean_acc)
                #     print('ave_test_abnormal_acc',ave_test_abnormal_acc)
                # with open('./decision_result.csv', 'a') as f:
                #     writer = csv.writer(f, delimiter=',', lineterminator='\n', )
                writer.writerow([use_hybrid,n_repetition,length,use_his,adap_pool,ave_test_acc,ave_ori_acc,\
                                 ave_argmax_acc,ave_mean_acc,use_session_or_patients,total,num_epochs,\
                                 time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time())),batch_size,\
                                 num_epochs,lr,fix_testset,start,rows,block,hidden_layers,hidden_length,ave_test_sen,ave_test_spe,ave_ori_sen,ave_ori_spe,ave_mean_sen,ave_mean_spe,con_matrix,con_matrix_mean,con_matrix_no,ave_mul_acc1,ave_mul_sen,ave_mul_spe])













