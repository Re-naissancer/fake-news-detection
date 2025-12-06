import os
import torch


class Config:
    # 路径设置
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, 'data/raw')
    MODEL_DIR = os.path.join(ROOT_DIR, 'pretrained_models')
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
    RESULT_DIR = os.path.join(ROOT_DIR, 'result')

    TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
    TEST_FILE = os.path.join(DATA_DIR, 'Atest.csv')
    SUB_FILE = os.path.join(RESULT_DIR, 'submission.csv')

    # 模型参数
    BERT_PATH = os.path.join(MODEL_DIR, 'chinese-macbert-large')
    # BERT_PATH = 'hfl/chinese-macbert-large'

    # 训练超参数
    SEED = 42
    MAX_LEN = 256
    BATCH_SIZE = 32
    EPOCHS = 4
    LEARNING_RATE = 1.5e-5
    EPS = 1e-8
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01

    # 进阶参数
    N_FOLDS = 5  # 5折交叉验证
    USE_PGD = True  # 开启 PGD 对抗训练
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)