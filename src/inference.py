import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

from .config import Config
from .dataset import FakeNewsDataset
from .model import FakeNewsModel


def inference():
    conf = Config()
    df_test = pd.read_csv(conf.TEST_FILE)

    test_dataset = FakeNewsDataset(df_test, conf, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=conf.BATCH_SIZE, shuffle=False, num_workers=2)

    # 存储最终结果，形状为 (样本数, 1)
    final_preds = np.zeros((len(df_test), 1))

    # 🌟 关键修改：计算实际训练出的模型数量 N 🌟
    trained_folds = 0

    # 遍历所有可能的 Fold
    for fold in range(conf.N_FOLDS):
        model_path = f"{conf.OUTPUT_DIR}/bert_lstm_fold{fold}.pth"

        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"Warning: Model for Fold {fold + 1} not found at {model_path}. Skipping.")
            continue  # 如果文件不存在，跳过当前循环

        print(f"Inference using Fold {fold + 1} model...")

        # 计数
        trained_folds += 1

        model = FakeNewsModel(conf)
        model.load_state_dict(torch.load(model_path, map_location=conf.DEVICE))
        model.to(conf.DEVICE)
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for data in tqdm(test_loader):
                ids = data['ids'].to(conf.DEVICE)
                mask = data['mask'].to(conf.DEVICE)
                token_type_ids = data['token_type_ids'].to(conf.DEVICE)

                outputs = model(ids, mask, token_type_ids)
                outputs = torch.sigmoid(outputs)  # 转为概率
                fold_preds.extend(outputs.cpu().numpy())

        # 将当前fold的预测结果加到总结果中
        final_preds += np.array(fold_preds)

    # 🌟 关键修改：使用实际训练出的模型数量进行平均 🌟
    if trained_folds == 0:
        print("Error: No trained models found. Cannot perform inference.")
        return


    # 取平均
    final_preds /= trained_folds
    final_preds = final_preds.flatten()

    # 生成提交文件
    submission = pd.DataFrame()
    submission['id'] = df_test['id']
    submission['prob'] = final_preds

    submission.to_csv(conf.SUB_FILE, index=False)
    print(f"Submission saved to {conf.SUB_FILE}")


if __name__ == "__main__":
    inference()