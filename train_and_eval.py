
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
import torch
from braindecode.datasets import TUHAbnormal,TUH,BaseConcatDataset
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.models import ShallowFBCSPNet, Deep4Net,EEGNetv4,EEGNetv1,EEGResNet,TCN,SleepStagerBlanco2020,USleep,\
                                TIDNet,get_output_shape,HybridNet, SleepStagerChambon2018
from braindecode.preprocessing import (
    exponential_moving_standardize, preprocess, Preprocessor, scale)
from braindecode.util import np_to_th
from braindecode.datautil import load_concat_dataset
from tcn_1 import TCN_1
from hybrid_1 import HybridNet_1

from train_and_eval_config import *

mne.set_log_level(mne_log_level)

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.benchmark = True  # Enables automatic algorithm optimizations
torch.set_num_threads(N_JOBS)  # Sets the available number of threads

data_loading_start = time.time()

def select_by_duration(ds, tmin=0, tmax=None):
    if tmax is None:
        tmax = np.inf
    # determine length of the recordings and select based on tmin and tmax
    split_ids = []
    for d_i, d in enumerate(ds.datasets):
        duration = d.raw.n_times / d.raw.info['sfreq']
        if tmin <= duration <= tmax:
            split_ids.append(d_i)
    splits = ds.split(split_ids)
    split = splits['0']
    return split

def session_key(file_name):
    """ sort the file name by session """
    return re.findall(r'(s\d{2})', file_name)

def natural_key(file_name):
    """ provides a human-like sorting key of a string """
    key = [int(token) if token.isdigit() else None
           for token in re.split(r'(\d+)', file_name)]
    return key

def time_key(file_name):
    """ provides a time-based sorting key """
    splits = file_name.split('/')
    [date] = re.findall(r'(\d{4}_\d{2}_\d{2})', splits[-1])
    date_id = [int(token) for token in date.split('_')]
    recording_id = natural_key(splits[-1])
    session_id = session_key(splits[-2])
    return date_id + session_id + recording_id

def read_all_file_names(path, extension, key="time"):
    """ read all files with specified extension from given path
    :param path: parent directory holding the files directly or in subdirectories
    :param extension: the type of the file, e.g. '.txt' or '.edf'
    :param key: the sorting of the files. natural e.g. 1, 2, 12, 21 (machine 1, 12, 2, 21) or by time since this is
    important for cv. time is specified in the edf file names
    """
    file_paths = glob.glob(path + '**/*' + extension, recursive=True)

    if key == 'time':
        return sorted(file_paths, key=time_key)

    elif key == 'natural':
        return sorted(file_paths, key=natural_key)

    else:
        return file_paths
def relabel(dataset,label_path,dataset_folder):
    des=dataset.description
    des_path=list(des['path'])
    des_file=[]
    for i in des_path:
        des_file.append(os.path.basename(i))
    # print(des_file)
    all_labelled_TUEG_file_names = []
    TUEG_labels = []
    with open(label_path, newline='') as csvfile:
        label_catalog_reader = csv.reader(csvfile, delimiter='\t')

        # Skip the header row (column names)
        next(label_catalog_reader, None)

        for row in label_catalog_reader:
            id, _ = os.path.splitext(os.path.basename(row[1]))

            p_ab = float(row[2])
            label_from_ML = row[3]
            label_from_rules = row[4]
            # if label_from_ML==label_from_rules and (p_ab>=0.99 or p_ab<=0.01):
            if (p_ab >= 0.99 or p_ab <= 0.01):
                label = label_from_ML
            else:
                continue
            full_folder = os.path.join(dataset_folder, row[0])
            this_file_names = read_all_file_names(full_folder, '.edf', key='time')
            # print(this_file_names)
            [all_labelled_TUEG_file_names.append(ff) for ff in this_file_names if (id in os.path.basename(ff) and os.path.basename(ff) in des_file)]
            [TUEG_labels.append(label) for ff in this_file_names if (id in os.path.basename(ff) and os.path.basename(ff) in des_file)]
    # print(all_labelled_TUEG_file_names)
    # print(list(des))
    if 'pathological' not in list(des):
        des['pathological']=[2]*len(des['age'])
    for i in range(len(all_labelled_TUEG_file_names)):
        des['pathological'][des_file.index(os.path.basename(all_labelled_TUEG_file_names[i]))]=bool(TUEG_labels[i])
    return des

# Remove unlabeled data
def select_labeled(ds):
    split_ids = []
    for d_i, d in enumerate(ds.description['pathological']):
        if d==True or d==False:
            split_ids.append(d_i)
    splits = ds.split(split_ids)
    split = splits['0']
    return split

def check_inf(ds):
    for d_i, d in enumerate(ds.datasets):
        print(d.raw.info)

# Ensure that each data has all required channels
def select_by_channel(ds, channels):
    split_ids = []
    for d_i, d in enumerate(ds.datasets):
        include=True
        for chan in channels:
            if chan in d.raw.info['ch_names']:
                continue
            else:
                include=False
                break
        if include:
            split_ids.append(d_i)
    splits = ds.split(split_ids)
    split = splits['0']
    return split

def custom_crop(raw, tmin=0.0, tmax=None, include_tmax=True):
    # crop recordings to tmin – tmax. can be incomplete if recording
    # has lower duration than tmax
    # by default mne fails if tmax is bigger than duration
    tmax = min((raw.n_times - 1) / raw.info['sfreq'], tmax)
    raw.crop(tmin=tmin, tmax=tmax, include_tmax=include_tmax)
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
        tueg_ids=list(range(n_tueg))
        ds_tuab= TUHAbnormal(
            tuab_path, recording_ids=tuab_ids,target_name='pathological',
            preload=preload)
        print(ds_tuab.description)
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
            # Preprocessor('crop',tmin=60,tmax=21*60),
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
        preprocess(ds, preprocessors)

        if saved_data:
            ds.save(saved_path,overwrite=True)

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
    train_set=splits['True']
    valid_set=splits['False']


n_channels = windows_ds[0][0].shape[0]

print("n_channels:",n_channels)
# n_times = windows_ds[0][0].shape[1]
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
    model=TCN_1(n_channels, n_classes, n_blocks=8, n_filters=2, kernel_size=11, drop_prob=0.2, add_log_softmax=False)
elif model_name=='hybridnet':
    model=HybridNet(n_channels,n_classes,window_len_samples)
elif model_name == 'hybridnet_1':
    model = HybridNet_1(n_channels, n_classes, window_len_samples)

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
from skorch.callbacks import Checkpoint

cp = Checkpoint(dirname='', f_criterion=None, f_optimizer=None, f_history=None)
clf = EEGClassifier(
    model,
    criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set) if test_on_eval else None,  # using valid_set for validation
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    callbacks=[
        "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),("cp",cp)
    ],
    device=device,
)
# Model training for a specified number of epochs. `y` is None as it is already supplied
# in the dataset.
clf.fit(train_set, y=None, epochs=n_epochs)
# clf.save_params('./params.pt')
model_training_end = time.time()

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# Extract loss and accuracy values for plotting from history object
results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,
                  index=clf.history[:, 'epoch'])

# get percent of misclass for better visual comparison to loss
df = df.assign(train_misclass=100 - 100 * df.train_accuracy,
               valid_misclass=100 - 100 * df.valid_accuracy)

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
y_true = test_set.get_metadata().target
y_pred = clf.predict(test_set)

# generating confusion matrix
confusion_mat = confusion_matrix(y_true, y_pred)
# print(confusion_mat)
# print(type(confusion_mat[0][0]))
# # add class labels
# # label_dict is class_name : str -> i_class : int
# label_dict = test_set.datasets[0].windows.event_id.items()
# print(label_dict)
# # # sort the labels by values (values are integer class labels)
# labels = list(dict(sorted(list(label_dict), key=lambda kv: kv[1])).keys())
# print(labels)
labels=['normal','abnormal']
# plot the basic conf. matrix
plot_confusion_matrix(confusion_mat, class_names=labels) #if there is something wrong, change the version of matplotlib to 3.0.3, or find the result in confusion_mat
# plot_confusion_matrix(confusion_mat)
plt.show()

