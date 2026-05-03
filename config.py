import torch


DEFOG_DIR = 'data/parkinsons-fog/trainData/defog/'
TDCSFOG_DIR = 'trainData/tdcsfog/'
LSTM_MODEL_PATH = 'best_lstm.pth'
RNN_MODEL_PATH = 'best_rnn.pth'
CNN_MODEL_PATH = 'best_cnn.pth'
LSTM_FOCAL_MODEL_PATH = 'best_lstm_focal.pth'
WINDOW_SIZE = 256
STEP_SIZE = 128
FEATURES = ['AccV', 'AccML', 'AccAP']
LABELS = ['StartHesitation', 'Turn', 'Walking']
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4
VAL_SUBJECTS = 0.2
RANDOM_SEED = 42
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps'  if torch.backends.mps.is_available() else
    'cpu'
)