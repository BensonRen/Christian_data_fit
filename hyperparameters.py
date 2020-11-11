"""
Parameter file for specifying the running parameters for forward model
"""
# Model Architectural Parameters
DATA_SET = 'meta_material'
# USE_LORENTZ = True
NUM_LOR = 6
# USE_CONV = False                         # Whether use upconv layer when not using lorentz @Omar
# LINEAR = [8, 50, 50]
# # If the Lorentzian is False
# CONV_OUT_CHANNEL = [4, 4, 4]
# CONV_KERNEL_SIZE = [8, 5, 5]
# CONV_STRIDE = [2, 1, 1]

# Optimization parameters
OPTIM = "Adam"
REG_SCALE = 0
BATCH_SIZE = 1
EVAL_STEP = 50
RECORD_STEP = 50
TRAIN_STEP = 100000
LEARN_RATE = 1e-1
# DECAY_STEP = 25000 # This is for step decay, however we are using dynamic decaying
LR_DECAY_RATE = 0.5
STOP_THRESHOLD = 0.001
USE_CLIP = False
GRAD_CLIP = 1
USE_WARM_RESTART = False
LR_WARM_RESTART = 600
ERR_EXP = 2
DELTA = 0
GRADIENT_ASCEND_STRENGTH = 10
OPTIMIZE_W0_RATIO = 0

# Data Specific parameters
X_RANGE = [i for i in range(2, 10 )]
Y_RANGE = [i for i in range(10 , 2011 )]
FREQ_LOW = 0.8
FREQ_HIGH = 1.5
NUM_SPEC_POINTS = 300
FORCE_RUN = True
DATA_DIR = ''                # For local usage
# DATA_DIR = '/work/sr365/Omar_data'
# DATA_DIR = 'C:/Users/labuser/mlmOK_Pytorch/'                # For Omar office desktop usage
# DATA_DIR = '/home/omar/PycharmProjects/mlmOK_Pytorch/'  # For Omar laptop usage
GEOBOUNDARY = [30, 52, 42, 52]
NORMALIZE_INPUT = True
TEST_RATIO = 0.001

# Running specific
USE_CPU_ONLY = False
MODEL_NAME = "Gradient_Ascend"
SPECTRA_SAVE_DIR = '/work/sr365/Christian_data_fit/num_lor' + str(NUM_LOR)
EVAL_MODEL = None
# PRE_TRAIN_MODEL = "Gradient_Ascend"
NUM_PLOT_COMPARE = 5
