import os
import torch

SEED = 42
KAGGLE_INPUT_DIR = 'input'
KAGGLE_WORKING_DIR = 'outputs'
MODEL_NAME = "morse_crnn_4layer_norm_aug_v1"

FILE_PATH_COLUMN = 'id'
TARGET_COLUMN = 'message'
TEST_ID_COLUMN = 'id'

BLANK_CHAR = '<blank>'

AUDIO = {
    "sample_rate": 8000, "n_mels": 64, "n_fft": 400,
    "hop_length": 160, "win_length": 400,
    "amplitude_to_db": True,
    "top_db": 80,
}

PREPROCESSING = {
    "normalize": {
        "apply": True,
        "target_dbfs": -23.0,
        "silence_threshold_db": -60
    },
    "frequency_filter": {
        "apply": True,
        "min_mel_bin": 12,
        "max_mel_bin": 40,
        "mask_value": None
    }
}

AUGMENTATION = {
    "probability": 0.9,
    "active": ['time_shift', 'gain', 'noise', 'pitch_shift', 'freq_masking'],
    "time_shift": {"max_fraction": 0.1},
    "gain": {"db_range": (-8.0, 4.0)},
    "noise": {"snr_db_range": (0, 20), "types_prob": {'gaussian': 0.6, 'pink': 0.4}},
    "pitch_shift": {"n_steps_range": (-1.5, 1.5)},
    "freq_masking": {"masks": 2, "width": 8}
}

TRAINING = {
    "batch_size": 128,
    "num_workers": 4,
    "learning_rate": 3e-4,
    "num_epochs": 40,
    "validation_split_size": 0.1,
    "optimizer": "AdamW",
    "optimizer_weight_decay": 1e-5,
    "scheduler": "ReduceLROnPlateau",
    "scheduler_patience": 5,
    "scheduler_factor": 0.5,
    "grad_clip_value": 1.0
}

MODEL = {
    "type": "CRNNModel_4Layer",
    "rnn_hidden_size": 256,
    "rnn_num_layers": 2,
    "cnn_dropout": 0.15,
    "rnn_dropout": 0.15
}

INFERENCE = {
    "batch_size": 64,
    "use_beam_search": True,
    "beam_search_beam_size": 15,
    "beam_search_lm_path": None,
    "beam_search_lexicon_path": None,
}

TRAIN_CSV_PATH = os.path.join(KAGGLE_INPUT_DIR, 'train.csv')
TEST_CSV_PATH = os.path.join(KAGGLE_INPUT_DIR, 'test.csv')
SAMPLE_SUBMISSION_PATH = os.path.join(KAGGLE_INPUT_DIR, 'sample_submission.csv')
AUDIO_BASE_PATH = os.path.join(KAGGLE_INPUT_DIR, 'morse_dataset', 'morse_dataset')
CHECKPOINT_DIR = os.path.join(KAGGLE_WORKING_DIR, f"checkpoints_{MODEL_NAME}")
LOG_DIR = os.path.join(KAGGLE_WORKING_DIR, "logs") # Отдельная папка для логов
LOG_FILE = os.path.join(LOG_DIR, f"training_log_{MODEL_NAME}.csv") # CSV удобнее для анализа
SUBMISSION_FILE = os.path.join(KAGGLE_WORKING_DIR, f"submission_{MODEL_NAME}.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

