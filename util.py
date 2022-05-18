from itertools import product
import time
import os

import csv
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import glob
import re
import pandas as pd

def find_all_zero(input):
    res=[]
    for i in range(len(input)):
        if input[i]==0:
            res.append(i)
    return res


def top1(lst):
    return max(lst, default='empty', key=lambda v: lst.count(v))
def con_mat(starts,b,c):
    b = b.tolist()
    TT = 0
    TF = 0
    FT = 0
    FF = 0

    begin = starts[0]
    if len(starts)>1:
        for end in starts[1:]:
            predict=c[begin:end].tolist()
            predict=top1(predict)
            if predict==True:
                if b[begin]==True:
                    TT+=1
                else:
                    TF+=1
            else:
                if b[begin] == True:
                    FT+=1
                else:
                    FF+=1

            # print(predict)
            begin=end
    predict=c[begin:].tolist()
    predict=top1(predict)
    if predict == True:
        if b[begin] == True:
            TT += 1
        else:
            TF += 1
    else:
        if b[begin] == True:
            FT += 1
        else:
            FF += 1
    # print(predict)
    # print(TT,TF,FT,FF)
    return np.array([[FF,TF],[FT,TT]])
def timecost(time_dutation):
    m, s = divmod(time_dutation, 60)
    h, m = divmod(m, 60)
    return "%dh:%dm:%ds" % (h, m, s)
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
            # Skip blank lines
            if len(row) == 0:
                continue
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