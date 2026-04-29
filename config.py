import torch


DEFOG_DIR       =  'trainData/defog/'
TDCSFOG_DIR     = 'trainData/tdcsfog/'
LSTM_MODEL_PATH = 'best_lstm_model.pth'
RNN_MODEL_PATH  = 'rnn_model.pth'

WINDOW_SIZE = 256
STEP_SIZE   = 128


FEATURES = ['AccV', 'AccML', 'AccAP']
LABELS   = ['StartHesitation', 'Turn', 'Walking']


BATCH_SIZE  = 64
EPOCHS      = 15
LEARNING_RATE = 1e-3
TEST_SIZE   = 0.2
RANDOM_SEED = 42


DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')