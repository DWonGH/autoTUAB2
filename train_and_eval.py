
from itertools import product
import time
import os
from torch import nn, optim
from torch.utils.data import DataLoader
import csv
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import glob
import re
from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer
import matplotlib.pyplot as plt
import torch
from braindecode.datasets import TUHAbnormal,TUH,BaseConcatDataset
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.models import ShallowFBCSPNet, Deep4Net,EEGNetv4,EEGNetv1,EEGResNet,TCN,SleepStagerBlanco2020,USleep,\
                                TIDNet,get_output_shape,HybridNet, SleepStagerChambon2018
from braindecode.preprocessing import (
    exponential_moving_standardize, preprocess, Preprocessor, scale)
from braindecode.util import np_to_th
from braindecode.datautil import load_concat_dataset
from deep4_1 import Deep4Net_1
from tcn_1 import TCN_1
from hybrid_1 import HybridNet_1
from vit import ViT

from util import *
from batch_test_hyperparameters import *


with open(log_path,'a') as f:
    writer=csv.writer(f, delimiter=',',lineterminator='\n',)
    writer.writerow([time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))])
    writer.writerow(['train_loss', 'valid_loss',  'train_accuracy',  'valid_accuracy','etl_time','model_training_time',\
     'test_acc','test_precision','test_recall',\
     'n_repetition','mne_log_level','random_state','tuab','tueg','n_tuab','n_tueg','n_load','preload','window_len_s',\
     'tuab_path','tueg_path','saved_data','saved_path','saved_windows_data','saved_windows_path',\
     'load_saved_data','load_saved_windows','bandpass_filter','low_cut_hz','high_cut_hz',\
     'standardization','factor_new','init_block_size','n_jobs','n_classes','lr','weight_decay',\
     'batch_size','n_epochs','tmin','tmax','multiple','sec_to_cut','duration_recording_sec','max_abs_val',\
     'sampling_freq','test_on_eval','split_way','train_size','valid_size','test_size','shuffle',\
     'model_name','final_conv_length','window_stride_samples','relabel_dataset','relabel_label',\
     'channels','drop_prob','n_blocks','n_filters', 'kernel_size','precision_per_recording','recall_per_recording','acc_per_recording'])

    # Iterate over data/preproc parameters
    for (mne_log_level,random_state,tuab,tueg,n_tuab,n_tueg,n_load,preload,window_len_s,\
         tuab_path,tueg_path,saved_data,saved_path,saved_windows_data,saved_windows_path,\
         load_saved_data,load_saved_windows,bandpass_filter,low_cut_hz,high_cut_hz,\
         standardization,factor_new,init_block_size,n_jobs,tmin,tmax,multiple,sec_to_cut,duration_recording_sec,max_abs_val,\
         sampling_freq,test_on_eval,split_way,train_size,valid_size,test_size,shuffle,window_stride_samples,\
         relabel_dataset,relabel_label,channels) in product(
                MNE_LOG_LEVEL,RANDOM_STATE,TUAB,TUEG,N_TUAB,N_TUEG,N_LOAD,PRELOAD,\
                WINDOW_LEN_S,TUAB_PATH,TUEG_PATH,SAVED_DATA,SAVED_PATH,SAVED_WINDOWS_DATA,\
                SAVED_WINDOWS_PATH,LOAD_SAVED_DATA,LOAD_SAVED_WINDOWS,BANDPASS_FILTER,\
                LOW_CUT_HZ,HIGH_CUT_HZ,STANDARDIZATION,FACTOR_NEW,INIT_BLOCK_SIZE,N_JOBS,\
                TMIN,TMAX,MULTIPLE,SEC_TO_CUT,\
                DURATION_RECORDING_SEC,MAX_ABS_VAL,SAMPLING_FREQ,TEST_ON_VAL,SPLIT_WAY,\
                TRAIN_SIZE,VALID_SIZE,TEST_SIZE,SHUFFLE,WINDOW_STRIDE_SAMPLES,RELABEL_DATASET,RELABEL_LABEL,CHANNELS):
        print(mne_log_level, random_state, tuab, tueg, n_tuab, n_tueg, n_load, preload, window_len_s, \
        tuab_path, tueg_path, saved_data, saved_path, saved_windows_data, saved_windows_path, \
        load_saved_data, load_saved_windows, bandpass_filter, low_cut_hz, high_cut_hz, \
        standardization, factor_new, init_block_size, n_jobs, \
        tmin, tmax, multiple, sec_to_cut, duration_recording_sec, max_abs_val, \
        sampling_freq, test_on_eval, split_way, train_size, valid_size, test_size, shuffle, \
        relabel_dataset, relabel_label, \
        channels)

        cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
        device = 'cuda' if cuda else 'cpu'
        if cuda:
            torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.benchmark = True  # Enables automatic algorithm optimizations
        torch.set_num_threads(n_jobs)  # Sets the available number of threads

        mne.set_log_level(mne_log_level)

        data_loading_start = time.time()


        window_len_samples = window_len_s*sampling_freq
        if load_saved_windows:
            load_ids = list(range(n_load))
            windows_ds = load_concat_dataset(
                path=saved_windows_path,
                preload=False,
                ids_to_load=load_ids,
                target_name='pathological',
            )
        else:
            if load_saved_data:
                load_ids=list(range(n_load))
                ds=load_concat_dataset(
                path=saved_path,
                preload=preload,
                ids_to_load=load_ids,
                target_name='pathological',
            )
            else:
                tuab_ids = list(range(n_tuab))
                ds_tuab= TUHAbnormal(
                    tuab_path, recording_ids=tuab_ids,target_name='pathological',
                    preload=preload)
                print(ds_tuab.description)

                if tueg:
                    tueg_ids=list(range(n_tueg))
                    ds_tueg=TUH(tueg_path,recording_ids=tueg_ids,target_name='pathological',
                        preload=preload)
                    print(ds_tueg.description)

                ds=BaseConcatDataset(([i for i in ds_tuab.datasets] if tuab else [])+([j for j in ds_tueg.datasets] if tueg else []))
                print(ds.description)
                ds=select_by_duration(ds,tmin,tmax)

                for i in range(len(relabel_label)):
                    ds.set_description(relabel(ds,relabel_label[i],relabel_dataset[i]),overwrite=True)
                print(ds.description)

                ds=select_labeled(ds)
                print(ds.description)

                ds=select_by_channel(ds,channels)
                # check_inf(ds)

                preprocessors = [
                    Preprocessor('pick_types', eeg=True, meg=False, stim=False),# Keep EEG sensors
                    Preprocessor('pick_channels',ch_names = channels),
                    Preprocessor(fn='resample', sfreq=sampling_freq),
                    Preprocessor(custom_crop, tmin=sec_to_cut, tmax=duration_recording_sec+sec_to_cut, include_tmax=False,
                                 apply_on_array=False),
                    # Preprocessor('crop',tmin=sec_to_cut,tmax=duration_recording_sec+sec_to_cut),
                    Preprocessor(scale, factor=1e6, apply_on_array=True),  # Convert from V to uV
                    # Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
                    Preprocessor(np.clip, a_min=-max_abs_val, a_max=max_abs_val, apply_on_array=True),
                    # Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                    #              factor_new=factor_new, init_block_size=init_block_size)
                ]
                if multiple:
                    preprocessors.append(Preprocessor(scale, factor=multiple,apply_on_array=True))
                if bandpass_filter:
                    preprocessors.append(Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz))
                if standardization:
                    preprocessors.append(Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                                 factor_new=factor_new, init_block_size=init_block_size))

                if preload:
                    preprocess(ds, preprocessors)
                    if saved_data:
                        ds.save(saved_path, overwrite=True)
                else:
                    preprocess(ds, preprocessors, n_jobs=n_jobs, save_dir=saved_path, overwrite=True)

            fs = ds.datasets[0].raw.info['sfreq']
            # print("fs:",fs)
            window_len_samples = int(fs * window_len_s)
            # window_stride_samples = int(fs * 4)
            if not window_stride_samples:
                window_stride_samples = window_len_samples


            # window_stride_samples = int(fs * window_len_s)
            windows_ds = create_fixed_length_windows(
                ds, start_offset_samples=0, stop_offset_samples=None,
                window_size_samples=window_len_samples,
                window_stride_samples=window_stride_samples, drop_last_window=True,
                preload=preload, drop_bad_windows=True)

            # Drop bad epochs
            # XXX: This could be parallelized.
            # XXX: Also, this could be implemented in the Dataset object itself.
            for ds in windows_ds.datasets:
                ds.windows.drop_bad()
                assert ds.windows.preload == preload

            if saved_windows_data:
                windows_ds.save(saved_windows_path,True)

        print(windows_ds.description)

        import torch
        from sklearn.model_selection import train_test_split

        if split_way=='proportion':
            idx_train, idx_valid_test = train_test_split(np.arange(len(windows_ds.description['path'])),
                                                    random_state=random_state,
                                                    train_size=train_size,
                                                    shuffle=shuffle)
            idx_valid,idx_test=train_test_split(idx_valid_test,random_state=random_state,test_size=test_size,shuffle=shuffle)
            splits = windows_ds.split(
                {"train": idx_train, "valid": idx_valid, "test": idx_test}
            )
            valid_set = splits["valid"]
            train_set = splits["train"]
            test_set = splits["test"]



            # valid_set = torch.utils.data.Subset(windows_ds, idx_valid)
            # train_set = torch.utils.data.Subset(windows_ds, idx_train)
            # test_set=torch.utils.data.Subset(windows_ds, idx_test)
        elif (split_way=='folder') :#this funtion do not create test_set now
            des=windows_ds.description
            if 'train' not in list(des):
                des['train']=[2]*len(des['path'])
            path=des['path']
            train=des['train']
            for i in range(len(train)):
                if train[i]!=True and train[i]!=False:
                    # print(train[i])
                    if 'train' in path[i]:
                        # print(path[i])
                        des['train']=True
                    elif 'eval' in path[i]:
                        des['train']=False
            windows_ds.set_description(des,overwrite=True)
            # print(windows_ds.description)
            splits=windows_ds.split('train')
            # print(splits)
            train_valid_set=splits['True']
            test_set=splits['False']
            idx_train, idx_valid = train_test_split(np.arange(len(train_valid_set.description['path'])),
                                                         random_state=random_state,
                                                         train_size=train_size,
                                                         shuffle=shuffle)
            splits = windows_ds.split(
                {"train": idx_train, "valid": idx_valid}
            )
            valid_set = splits["valid"]
            train_set = splits["train"]

        etl_time = time.time() - data_loading_start

        n_channels = windows_ds[0][0].shape[0]

        print("n_channels:",n_channels)
        # n_times = windows_ds[0][0].shape[1]


        # Iterate over model/training hyperparameters
        for (i, n_classes, lr, weight_decay, batch_size, n_epochs, model_name, final_conv_length,model_and_hpara) \
          in product(range(N_REPETITIONS), N_CLASSES, LR, WEIGHT_DECAY, BATCH_SIZE, N_EPOCHS, MODEL_NAME, \
          FINAL_CONV_LENGTH,MODEL_AND_HPARA):
            print(i, mne_log_level, random_state, tuab, tueg, n_tuab, n_tueg, n_load, preload, window_len_s, \
                  tuab_path, tueg_path, saved_data, saved_path, saved_windows_data, saved_windows_path, \
                  load_saved_data, load_saved_windows, bandpass_filter, low_cut_hz, high_cut_hz, \
                  standardization, factor_new, init_block_size, n_jobs, n_classes, lr, weight_decay, \
                  batch_size, n_epochs, tmin, tmax, multiple, sec_to_cut, duration_recording_sec, max_abs_val, \
                  sampling_freq, test_on_eval, split_way, train_size, valid_size, test_size, shuffle, \
                  model_name, final_conv_length, window_stride_samples, relabel_dataset, relabel_label, \
                  channels)
            # print(model_name)
            # model_name=model_and_hpara['model_name']
            # print(model_name)
            hpara=model_and_hpara['hpara']

            mne.set_log_level(mne_log_level)
            def exp(drop_prob=0.2,n_blocks=8, n_filters=2, kernel_size=11):
                n_blocks=int(n_blocks)
                n_filters=int(n_filters)
                kernel_size=int(kernel_size)
                print(drop_prob,n_blocks, n_filters, kernel_size)
                if model_name=='deep4':
                    model = Deep4Net(
                                n_channels, n_classes, input_window_samples=window_len_samples,
                                final_conv_length=final_conv_length, n_filters_time=25, n_filters_spat=25,
                                filter_time_length=10, pool_time_length=3, pool_time_stride=3,
                                n_filters_2=50, filter_length_2=10, n_filters_3=100,
                                filter_length_3=10, n_filters_4=200, filter_length_4=10,
                                first_pool_mode="max", later_pool_mode="max", drop_prob=0.5,
                                double_time_convs=False, split_first_layer=True, batch_norm=True,
                                batch_norm_alpha=0.1, stride_before_pool=False)
                elif model_name=='deep4_1':
                    model=Deep4Net_1(n_channels, n_classes, input_window_samples=window_len_samples,
                                final_conv_length=final_conv_length, n_filters_time=25, n_filters_spat=25,
                                filter_time_length=10, pool_time_length=3, pool_time_stride=3,
                                n_filters_2=50, filter_length_2=10, n_filters_3=100,
                                filter_length_3=10, n_filters_4=200, filter_length_4=10,
                                first_pool_mode="max", later_pool_mode="max", drop_prob=0.5,
                                double_time_convs=False, split_first_layer=True, batch_norm=True,
                                batch_norm_alpha=0.1, stride_before_pool=False,hpara=hpara)
                elif model_name=='shallow_smac':
                    model = ShallowFBCSPNet(
                        n_channels, n_classes, input_window_samples=window_len_samples,
                        n_filters_time=40, filter_time_length=25, n_filters_spat=40,
                        pool_time_length=75, pool_time_stride=15, final_conv_length=final_conv_length,
                        split_first_layer=True, batch_norm=True, batch_norm_alpha=0.1,
                        drop_prob=0.5)
                elif model_name=='eegnetv4':
                    model=EEGNetv4(n_channels, n_classes, input_window_samples=window_len_samples, final_conv_length=final_conv_length,
                                                pool_mode='mean', F1=8, D=2, F2=16, kernel_length=64, third_kernel_size=(8, 4),
                                                drop_prob=0.25)
                elif model_name=='eegnetv1':
                    model=EEGNetv1(n_channels, n_classes, input_window_samples=window_len_samples, final_conv_length=final_conv_length, pool_mode='max', second_kernel_size=(2, 32), third_kernel_size=(8, 4), drop_prob=0.25)
                elif model_name=='eegresnet':
                    model=EEGResNet(n_channels, n_classes, window_len_samples, final_conv_length, n_first_filters=10, n_layers_per_block=2, first_filter_length=3, split_first_layer=True, batch_norm_alpha=0.1, batch_norm_epsilon=0.0001)
                elif model_name=='tcn':
                    model=TCN(n_channels, n_classes, n_blocks=8, n_filters=2, kernel_size=12, drop_prob=0.2, add_log_softmax=False)
                elif model_name=='sleep2020':
                    model=SleepStagerBlanco2020(n_channels, sampling_freq, n_conv_chans=20, input_size_s=60, n_classes=2, n_groups=3, max_pool_size=2, dropout=0.5, apply_batch_norm=False, return_feats=False)
                elif model_name=='sleep2018':
                    model=SleepStagerChambon2018(n_channels, sampling_freq, n_conv_chs=8, time_conv_size_s=0.5, max_pool_size_s=0.125, pad_size_s=0.25, input_size_s=60, n_classes=2, dropout=0.25, apply_batch_norm=False, return_feats=False)
                elif model_name=='usleep':
                    model=USleep(in_chans=n_channels, sfreq=sampling_freq, depth=12, n_time_filters=5, complexity_factor=1.67, with_skip_connection=True, n_classes=2, input_size_s=60, time_conv_size_s=0.0703125, ensure_odd_conv_size=False, apply_softmax=False)
                elif model_name=='tidnet':
                    model=TIDNet(n_channels, n_classes, window_len_samples, s_growth=24, t_filters=32, drop_prob=0.4, pooling=15, temp_layers=2, spat_layers=2, temp_span=0.05, bottleneck=3, summary=- 1)
                elif model_name=='tcn_1':
                    model=TCN_1(n_channels, n_classes, n_blocks=n_blocks, n_filters=n_filters, kernel_size=kernel_size, drop_prob=drop_prob, add_log_softmax=True,input_window_samples=window_len_samples,last_layer_type='max_pool')
                elif model_name=='hybridnet':
                    model=HybridNet(n_channels,n_classes,window_len_samples)
                elif model_name == 'hybridnet_1':
                    model = HybridNet_1(n_channels, n_classes, window_len_samples)
                elif model_name == 'vit':
                    model = ViT(num_channels=n_channels,input_window_samples = window_len_samples,patch_size = 100,num_classes = n_classes,dim = 128,depth = 6,heads = 16,mlp_dim = 128,dropout = 0.1,emb_dropout = 0.1)

                print(get_output_shape(model,n_channels,window_len_samples))
                print(model)


                if cuda:
                    model.cuda()
                training_setup_end = time.time()

                # Start training loop
                model_training_start = time.time()

                from skorch.callbacks import LRScheduler
                from skorch.helper import predefined_split
                from braindecode import EEGClassifier
                from skorch.callbacks import Checkpoint,EarlyStopping

                monitor = lambda net: all(net.history[-1, ('train_loss_best', 'valid_loss_best')])
                cp = Checkpoint(monitor=monitor,dirname='', f_criterion=None, f_optimizer=None, load_best=False)
                callbacks=["accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),("cp",cp)]
                if earlystopping:
                    es=EarlyStopping()
                    callbacks.append(('es',es))
                clf = EEGClassifier(
                    model,
                    criterion=torch.nn.NLLLoss,
                    optimizer=torch.optim.AdamW,
                    train_split=predefined_split(valid_set) if test_on_eval else None,  # using valid_set for validation
                    optimizer__lr=lr,
                    optimizer__weight_decay=weight_decay,
                    batch_size=batch_size,
                    # callbacks=[
                    #     "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),("cp",cp),\
                    #     # ("es",es)
                    # ],
                    callbacks=callbacks,
                    device=device,
                )
                # Model training for a specified number of epochs. `y` is None as it is already supplied
                # in the dataset.
                clf.fit(train_set, y=None, epochs=n_epochs)
                # clf.save_params('./params.pt')
                model_training_time = time.time() - model_training_start

                import matplotlib.pyplot as plt
                from matplotlib.lines import Line2D

                # Extract loss and accuracy values for plotting from history object
                results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
                # print(clf.history)
                df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,
                                  index=clf.history[:, 'epoch'])
                # get percent of misclass for better visual comparison to loss
                df = df.assign(train_misclass=100 - 100 * df.train_accuracy,
                               valid_misclass=100 - 100 * df.valid_accuracy)
                print(df)
                if plot_result:
                    plt.style.use('seaborn')
                    fig, ax1 = plt.subplots(figsize=(8, 3))
                    df.loc[:, ['train_loss', 'valid_loss']].plot(
                        ax=ax1, style=['-', ':'], marker='o', color='tab:blue', legend=False, fontsize=14)

                    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
                    ax1.set_ylabel("Loss", color='tab:blue', fontsize=14)

                    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

                    df.loc[:, ['train_misclass', 'valid_misclass']].plot(
                        ax=ax2, style=['-', ':'], marker='o', color='tab:red', legend=False)
                    ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
                    ax2.set_ylabel("Misclassification Rate [%]", color='tab:red', fontsize=14)
                    ax2.set_ylim(ax2.get_ylim()[0], 85)  # make some room for legend
                    ax1.set_xlabel("Epoch", fontsize=14)

                    # where some data has already been plotted to ax
                    handles = []
                    handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Train'))
                    handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle=':', label='Valid'))
                    plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
                    plt.tight_layout()
                    plt.show()
                from sklearn.metrics import confusion_matrix
                from braindecode.visualization import plot_confusion_matrix

                # generate confusion matrices
                # print(test_set.description)
                y_true = test_set.get_metadata().target
                # print(y_true)
                starts=find_all_zero(test_set.get_metadata()['i_window_in_trial'].tolist())
                # print(starts)
                # print('y_true:',y_true)
                # print(len(y_true))
                y_pred = clf.predict(test_set)
                y_pred_proba=clf.predict_proba(test_set)




                # print(y_pred)
                # print('y_pred:',y_pred)
                # print(len(y_pred))
                confusion_mat_per_recording=con_mat(starts,y_true,y_pred)
                confusion_mat_per_recording_proba=con_mat(starts,y_true,y_pred,True,y_pred_proba)
                print(confusion_mat_per_recording)
                print(confusion_mat_per_recording_proba)


                # generating confusion matrix
                confusion_mat = confusion_matrix(y_true, y_pred)
                print(confusion_mat)
                # print(type(confusion_mat))
                precision=confusion_mat[0,0]/(confusion_mat[0,0]+confusion_mat[1,0])
                recall=confusion_mat[0,0]/(confusion_mat[0,0]+confusion_mat[0,1])
                acc=(confusion_mat[0,0]+confusion_mat[1,1])/(confusion_mat[0,0]+confusion_mat[0,1]+confusion_mat[1,1]+confusion_mat[1,0])
                precision_per_recording=confusion_mat_per_recording[0,0]/(confusion_mat_per_recording[0,0]+confusion_mat_per_recording[1,0])
                recall_per_recording=confusion_mat_per_recording[0,0]/(confusion_mat_per_recording[0,0]+confusion_mat_per_recording[0,1])
                acc_per_recording=(confusion_mat_per_recording[0,0]+confusion_mat_per_recording[1,1])/(confusion_mat_per_recording[0,0]+confusion_mat_per_recording[0,1]+confusion_mat_per_recording[1,1]+confusion_mat_per_recording[1,0])
                end=time.time()
                print('precision:',precision)
                print('recall:',recall)
                print('acc:',acc)
                print('precision_per_recording:', precision_per_recording)
                print('recall_per_recording:', recall_per_recording)
                print('acc_per_recording:', acc_per_recording)
                print('etl_time:',etl_time)
                print('model_training_time:',model_training_time)
                his_len=len(df)
                for i in range(his_len-1):
                    writer.writerow([df.loc[i+1][0],df.loc[i+1][1],df.loc[i+1][2],df.loc[i+1][3]])
                writer.writerow([df.loc[his_len][0],df.loc[his_len][1],df.loc[his_len][2],df.loc[his_len][3],etl_time,\
                 model_training_time,acc,precision,recall,i,mne_log_level,random_state,tuab,tueg,n_tuab,n_tueg,n_load,preload,\
                 window_len_s,tuab_path,tueg_path,saved_data,saved_path,saved_windows_data,saved_windows_path,\
                 load_saved_data,load_saved_windows,bandpass_filter,low_cut_hz,high_cut_hz,\
                 standardization,factor_new,init_block_size,n_jobs,n_classes,lr,weight_decay,\
                 batch_size,n_epochs,tmin,tmax,multiple,sec_to_cut,duration_recording_sec,max_abs_val,\
                 sampling_freq,test_on_eval,split_way,train_size,valid_size,test_size,shuffle,\
                 model_name,final_conv_length,window_stride_samples,relabel_dataset,relabel_label,\
                 channels,drop_prob,n_blocks, n_filters, kernel_size,precision_per_recording,recall_per_recording,acc_per_recording])
                # print(type(confusion_mat[0][0]))
                # # add class labels
                # # label_dict is class_name : str -> i_class : int
                # label_dict = test_set.datasets[0].windows.event_id.items()
                # print(label_dict)
                # # # sort the labels by values (values are integer class labels)
                # labels = list(dict(sorted(list(label_dict), key=lambda kv: kv[1])).keys())
                # print(labels)
                if plot_result:
                    labels=['normal','abnormal']
                    # plot the basic conf. matrix
                    plot_confusion_matrix(confusion_mat, class_names=labels) #if there is something wrong, change the version of matplotlib to 3.0.3, or find the result in confusion_mat
                    # plot_confusion_matrix(confusion_mat)
                    plt.show()
                    plot_confusion_matrix(confusion_mat_per_recording, class_names=labels)
                    plt.show()

                return acc

            if BO:
                bounds_transformer = SequentialDomainReductionTransformer()
                pbounds = {'drop_prob': (0,1),'n_blocks':(8,8.1), 'n_filters':(2,2.1), 'kernel_size':(11,11.1)}
                mutating_optimizer = BayesianOptimization(
                    f=exp,
                    pbounds=pbounds,
                    verbose=0,
                    random_state=1,
                    bounds_transformer=bounds_transformer
                )
                mutating_optimizer.maximize(
                    init_points=0,
                    n_iter=1,
                )
            else:
                exp()
