
log_path="result.csv"
plot_result=False
BO=False

MNE_LOG_LEVEL = ['WARNING' ] # avoid messages everytime a window is extracted

# parameters
N_REPETITIONS=1
RANDOM_STATE = [87]
TUAB=[True]
TUEG=[True]
N_TUAB=[50]
N_TUEG=[50]
N_LOAD=[50]
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
STANDARDIZATION=[False]
FACTOR_NEW = [1e-3]
INIT_BLOCK_SIZE = [1000]

N_JOBS = [8]
N_CLASSES = [2]
LR = [0.001]
WEIGHT_DECAY = [0.5 * 0.001]
BATCH_SIZE = [64]
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
MODEL_NAME = ['hybridnet_1']#Currently available:'deep4','eegnetv4','eegnetv1','sleep2020','usleep','tidnet','tcn_1',\
                        # 'hybridnet_1','eegresnet', 'vit'
FINAL_CONV_LENGTH = ["auto"]
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