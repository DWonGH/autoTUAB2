# This file sets various parameters used in train_and_eval.py.


log_path = "result.csv"
plot_result = False
BO = False
earlystopping = True
es_patience = 10
train_whole_dataset_again=True
test_model=False
params_deep4_60=['deep42022-08-22_08-28-46params.pt','deep42022-08-22_13-33-59params.pt','deep42022-08-22_20-08-07params.pt','deep42022-08-23_02-52-32params.pt','deep42022-08-23_07-31-26params.pt']
params_deep4_600=['deep42022-08-23_12-11-41params.pt','deep42022-08-23_15-52-28params.pt','deep42022-08-23_16-51-15params.pt','deep42022-08-24_01-08-18params.pt','deep42022-08-24_05-00-21params.pt','deep42022-08-24_10-58-14params.pt','deep42022-08-24_11-04-58params.pt']
params_deep4_300=['deep42022-08-25_19-30-09params.pt','deep42022-08-26_01-36-44params.pt','deep42022-08-26_06-51-30params.pt','deep42022-08-26_12-31-27params.pt','deep42022-08-26_18-57-53params.pt','deep42022-08-26_20-46-02params.pt','deep42022-08-27_04-37-34params.pt']
params_deep4_180=['deep42022-08-27_10-45-50params.pt','deep42022-08-27_16-24-18params.pt','deep42022-08-27_20-36-19params.pt','deep42022-08-28_02-09-57params.pt','deep42022-08-28_06-06-26params.pt','deep42022-08-28_10-17-49params.pt']
params_tcn1_900=['tcn_12022-08-30_15-53-05params.pt','tcn_12022-08-30_22-48-56params.pt','tcn_12022-08-31_05-29-42params.pt','tcn_12022-08-31_10-36-47params.pt','tcn_12022-08-31_17-31-53params.pt']
params_tcn1_300=['tcn_12022-09-04_11-55-13params.pt','tcn_12022-09-04_22-29-55params.pt','tcn_12022-09-05_09-25-52params.pt','tcn_12022-09-05_20-26-47params.pt','tcn_12022-09-06_07-21-07params.pt']
params_tcn1_180=['tcn_12022-09-06_22-35-51params.pt','tcn_12022-09-07_13-45-03params.pt','tcn_12022-09-08_06-01-55params.pt','tcn_12022-09-08_16-48-06params.pt','tcn_12022-09-10_02-01-38params.pt','tcn_12022-09-10_13-57-31params.pt','tcn_12022-09-10_17-10-35params.pt']
params_tcn1_60=['tcn_12022-09-12_07-34-49params.pt','tcn_12022-09-15_14-31-46params.pt','tcn_12022-09-16_15-11-31params.pt','tcn_12022-09-18_18-09-03params.pt','tcn_12022-09-20_05-59-27params.pt']


# Set verbosity of outputs from MNE library.
# Options: DEBUG, INFO, WARNING, ERROR, or CRITICAL.
# WARNING or higher avoids messages printing every time a window is extracted.
mne_log_level = 'ERROR'

# # parameters
# random_state = 87
# tuab=True
# tueg=True
# n_tuab=2991
# n_tueg=50
# n_load=2909
# preload=False
# window_len_s=60
# tuab_path = 'G:\\TUAB_relabelled\\v2.0.0\\edf'
# tueg_path = 'G:\\TUEG'
# saved_data=False # N.B. If preload=False then data will be saved after pre-processing, even if saved_data=False
# saved_path='D:\\autotuab2\\saved_data'
# saved_windows_data=False
# saved_windows_path='D:\\autotuab2\\saved_windows_data'
# load_saved_data=True
# load_saved_windows=False
# bandpass_filter=False
# low_cut_hz = 4.  # low cut frequency for filtering
# high_cut_hz = 38.  # high cut frequency for filtering
#
# # Parameters for exponential moving standardization
# standardization=True
# factor_new = 1e-3
# init_block_size = 1000
#
# n_jobs = 8
# n_classes = 2
# lr = 0.001
# weight_decay = 0.5 * 0.001
# batch_size = 64
# n_epochs = 10
# # determine length of the recordings and select based on tmin and tmax
# tmin = 5 * 60
# tmax = 35* 60
# multiple=0
# sec_to_cut = 60  # cut away at start of each recording
# duration_recording_sec = 20*60  # how many minutes to use per recording
# max_abs_val = 800  # for clipping
# sampling_freq = 100
# test_on_eval = True  # test on evaluation set or on training set
# split_way='folder' #'proportion' or 'folder'
# train_size=0.6 #train_size+valid_size+test_size=1.0
# valid_size=0.2
# test_size=0.2
# shuffle = True
# model_name = 'tcn_1'#Currently available:'deep4','eegnetv4','eegnetv1','sleep2020','usleep','tidnet','tcn_1',\
#                         # 'hybridnet_1','eegresnet'
# window_len_samples = window_len_s*sampling_freq  #this parameter is calculated automatically
# final_conv_length = "auto"
# window_stride_samples=None #if None, window_stride_samples = window_len_samples
# #The next two parameters can be extended. For example, [dataset1,dataset2,...] [label1,label2,...]
# relabel_dataset=['G:\\TUEG']
# relabel_label=['C:\\Users\\dg-western\\OneDrive - UWE Bristol (Staff)\\Academic\\Projects\\Quetzal\\Code\\autoTUAB2\\tueg_labels.csv']
#
# channels=[
#             'EEG A1-REF', 'EEG A2-REF',
#             'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
#             'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
#             'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
#             'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF']

test_on_brainvision = False
brainvision_path = 'D:\\phd\\sleep\\data\\Fastball'

# model specific hyperparameters

# tcn
tcn_kernel_size = 11
tcn_n_blocks = 5 # 8 from Bai. 5 from Gemein et al.
tcn_n_filters = 55 # Was 2. Gemein et al said they used 55 'channels' for each block.
tcn_add_log_softmax = True
tcn_last_layer_type = 'max_pool'
tcn_dropout = 0.05270154233150525

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
vit_dim = 64
vit_depth = 6
vit_heads = 16
vit_mlp_dim = 64
vit_emb_dropout = 0.1
