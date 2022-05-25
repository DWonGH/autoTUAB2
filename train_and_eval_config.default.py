# This file sets various parameters used in train_and_eval.py.

log_path = "result.csv"
plot_result = True
BO = False
earlystopping = True
es_patience = 10

# Set verbosity of outputs from MNE library.
# Options: DEBUG, INFO, WARNING, ERROR, or CRITICAL.
# WARNING or higher avoids messages printing every time a window is extracted.
mne_log_level = 'WARNING'

# parameters
random_state = 87
tuab=True
tueg=True
n_tuab=50
n_tueg=50
n_load=50
preload=True
window_len_s=60
tuab_path = 'D:/phd/tuab3g/v2.0.0/edf'
tueg_path = 'D:/phd/tueg1g'
saved_data=False
saved_path='D:\\phd\\saved_data'
saved_windows_data=False
saved_windows_path='D:\\phd\\saved_windows_data'
load_saved_data=False
load_saved_windows=False
bandpass_filter=False
low_cut_hz = 4.  # low cut frequency for filtering
high_cut_hz = 38.  # high cut frequency for filtering

# Parameters for exponential moving standardization
standardization=False
factor_new = 1e-3
init_block_size = 1000

n_jobs = 8
n_classes = 2
lr = 0.001
weight_decay = 0.5 * 0.001
batch_size = 64
n_epochs = 2
# determine length of the recordings and select based on tmin and tmax
tmin = 5 * 60
tmax = 35* 60
multiple=0
sec_to_cut = 60  # cut away at start of each recording
duration_recording_sec = 20*60  # how many minutes to use per recording
max_abs_val = 800  # for clipping
sampling_freq = 100
test_on_eval = True  # test on evaluation set or on training set
split_way='proportion' #'proportion' or 'folder'
train_size=0.6 #train_size+valid_size+test_size=1.0
valid_size=0.2
test_size=0.2
shuffle = True
model_name = 'hybridnet_1'#Currently available:'deep4','eegnetv4','eegnetv1','sleep2020','usleep','tidnet','tcn_1',\
                        # 'hybridnet_1','eegresnet'
window_len_samples = window_len_s*sampling_freq  #this parameter is calculated automatically
final_conv_length = "auto"
window_stride_samples=None #if None, window_stride_samples = window_len_samples
#The next two parameters can be extended. For example, [dataset1,dataset2,...] [label1,label2,...]
relabel_dataset=['D:/phd/tueg1g']
relabel_label=['D:\\phd\\autoTUAB2\\tueg_labels.csv']

channels=[
            'EEG A1-REF', 'EEG A2-REF',
            'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
            'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
            'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
            'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF']

# model specific hyperparameters

# tcn
tcn_kernel_size=11
tcn_n_blocks=8
tcn_n_filters=2
tcn_add_log_softmax=True
tcn_last_layer_type='max_pool'

deep4_n_filters_time=25
deep4_n_filters_spat=25
deep4_filter_time_length=10
deep4_pool_time_length=3
deep4_pool_time_stride=3
deep4_n_filters_2=50
deep4_filter_length_2=10
deep4_n_filters_3=100
deep4_filter_length_3=10
deep4_n_filters_4=200
deep4_filter_length_4=10
deep4_first_pool_mode="max"
deep4_later_pool_mode="max"
deep4_double_time_convs=False
deep4_split_first_layer=True
deep4_batch_norm=True
deep4_batch_norm_alpha=0.1
deep4_stride_before_pool=False

shallow_n_filters_time=40
shallow_filter_time_length=25
shallow_n_filters_spat=40
shallow_pool_time_length=75
shallow_pool_time_stride=15
shallow_split_first_layer=True
shallow_batch_norm=True
shallow_batch_norm_alpha=0.1

vit_patch_size = 10
vit_dim = 128
vit_depth = 6
vit_heads = 16
vit_mlp_dim = 128
vit_emb_dropout = 0.1
