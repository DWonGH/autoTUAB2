
mne_log_level = 'WARNING'  # avoid messages everytime a window is extracted

# parameters
random_state = 87
tuab=True
tueg=True
n_tuab=50
n_tueg=50
n_load=50
preload=True
window_len_s=60
tuab_path = 'G:\\TUAB\\v2.0.0\\edf'
tueg_path = 'G:\\TUEG'
saved_data=False # N.B. If preload=False then data will be saved after pre-processing, even if saved_data=True
saved_path='D:\\autotuab2\\saved_data'
saved_windows_data=False
saved_windows_path='D:\\autotuab2\\saved_windows_data'
load_saved_data=False
load_saved_windows=False
bandpass_filter=False
low_cut_hz = 4.  # low cut frequency for filtering
high_cut_hz = 38.  # high cut frequency for filtering

# Parameters for exponential moving standardization
standardization=True
factor_new = 1e-3
init_block_size = 1000

N_JOBS = 8
n_classes = 2
lr = 0.001
weight_decay = 0.5 * 0.001
batch_size = 64
n_epochs = 10
# determine length of the recordings and select based on tmin and tmax
tmin = 5 * 60
tmax = 35* 60
multiple=0
sec_to_cut = 60  # cut away at start of each recording
duration_recording_sec = 20*60  # how many minutes to use per recording
max_abs_val = 800  # for clipping
sampling_freq = 100
test_on_eval = True  # test on evaluation set or on training set
split_way='folder' #'proportion' or 'folder'
train_size=0.6 #train_size+valid_size+test_size=1.0
valid_size=0.2
test_size=0.2
shuffle = True
model_name = 'deep4'#Currently available:'deep4','eegnetv4','eegnetv1','sleep2020','usleep','tidnet','tcn_1',\
                        # 'hybridnet_1','eegresnet'
n_start_chans = 25
window_len_samples = window_len_s*sampling_freq  #this parameter is calculated automatically
final_conv_length = "auto"
init_lr = 1e-3 #This parameter is set differently in the old version and demo
window_stride_samples=None #if None, window_stride_samples = window_len_samples
#The next two parameters can be extended. For example, [dataset1,dataset2,...] [label1,label2,...]
relabel_dataset=['D:/phd/tueg1g']
relabel_label=['D:\\phd\\autotuab\\tueg_labels.csv']

channels=[
            'EEG A1-REF', 'EEG A2-REF',
            'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
            'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
            'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
            'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF']