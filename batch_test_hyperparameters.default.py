# parameters

N_REPETITIONS=1
RANDOM_STATE = [87]
TUAB=[True]
TUEG=[True]

# Setting any of the following to None will use all available recordings in the corresponding category.
# e.g. N_TUAB = None will use all recordings in the TUAB dataset.
N_TUAB=[50]
N_TUEG=[50]
N_LOAD=[25]

PRELOAD=[True]
WINDOW_LEN_S=[60]
TUAB_PATH = ['D:/phd/tuab3g/v2.0.0/edf']
TUEG_PATH = ['D:/phd/tueg1g']
SAVED_DATA=[False]
SAVED_PATH=['D:\\phd\\saved_data']
SAVED_WINDOWS_DATA=[False]
SAVED_WINDOWS_PATH=['D:\\phd\\saved_windows_data']
LOAD_SAVED_DATA=[False]
LOAD_SAVED_WINDOWS=[False]
BANDPASS_FILTER=[False]
LOW_CUT_HZ = [4. ] # low cut frequency for filtering
HIGH_CUT_HZ = [38.  ]# high cut frequency for filtering

# Parameters for exponential moving standardization
STANDARDIZATION=[True]
FACTOR_NEW = [1e-3]
INIT_BLOCK_SIZE = [1000]

N_JOBS = [8]
N_CLASSES = [2]
LR = [0.001]
WEIGHT_DECAY = [0.5 * 0.001]
BATCH_SIZE = [1]
N_EPOCHS = [2]
# determine length of the recordings and select based on tmin and tmax
TMIN = [5 * 60]
TMAX = [35* 60]
MULTIPLE=[0]
SEC_TO_CUT = [60]  # cut away at start of each recording
DURATION_RECORDING_SEC =[20*60 ] # how many minutes to use per recording
MAX_ABS_VAL =[800]  # for clipping
SAMPLING_FREQ = [100]
TEST_ON_VAL = [True]  # test on evaluation set or on training set
SPLIT_WAY=['proportion'] #'proportion' or 'folder'
TRAIN_SIZE=[0.6 ]#train_size+valid_size+test_size=1.0
VALID_SIZE=[0.2]
TEST_SIZE=[0.2]
SHUFFLE = [True]
MODEL_NAME = ['vit']#Currently available:'deep4','eegnetv4','eegnetv1','sleep2020','usleep','tidnet','tcn_1',\
                        # 'hybridnet_1','eegresnet','vit'
# model specific hyperparameters
DEEP4_BATCH_NORM_ALPHA=[0.1,0.2]

FINAL_CONV_LENGTH = ["auto"]
DROPOUT=[0.1]
WINDOW_STRIDE_SAMPLES=[None] #if None, window_stride_samples = window_len_samples
#The next two parameters can be extended. For example, [dataset1,dataset2,...] [label1,label2,...]
RELABEL_DATASET=[['D:/phd/tueg1g']]
RELABEL_LABEL=[['D:\\phd\\autoTUAB2\\tueg_labels.csv']]

CHANNELS=[[
            'EEG A1-REF', 'EEG A2-REF',
            'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
            'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
            'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
            'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF']]