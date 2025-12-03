import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
import tqdm

from .config import Config
from .dataset import FakeNewsDataset
from .model import FakeNewsModel
from .utils import seed_everything, calculate_auc, PGD


# def train_fn(model, train_loader, optimizer, scheduler, device, criterion, fgm=None):
#     model.train()
#     losses = []
#
#     for data in tqdm.tqdm(train_loader, desc="Training"):
#         ids = data['ids'].to(device)
#         mask = data['mask'].to(device)
#         token_type_ids = data['token_type_ids'].to(device)
#         labels = data['labels'].to(device).view(-1, 1)
#
#         optimizer.zero_grad()
#         outputs = model(ids, mask, token_type_ids)
#         loss = criterion(outputs, labels)
#         loss.backward()
#
#         # 对抗训练
#         if fgm is not None:
#             fgm.attack()  # 攻击
#             outputs_adv = model(ids, mask, token_type_ids)
#             loss_adv = criterion(outputs_adv, labels)
#             loss_adv.backward()
#             fgm.restore()  # 恢复
#
#         optimizer.step()
#         scheduler.step()
#         losses.append(loss.item())
#
#     return np.mean(losses)


def train_fn(model, train_loader, optimizer, scheduler, device, criterion, pgd_attacker):
    model.train()
    losses = []

    # PGD 参数设置
    K = 3  # 攻击次数

    for data in tqdm.tqdm(train_loader, desc="Training"):
        ids = data['ids'].to(device)
        mask = data['mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        labels = data['labels'].to(device).view(-1, 1)

        # 1. 正常前向传播
        outputs = model(ids, mask, token_type_ids)
        loss = criterion(outputs, labels)
        loss.backward()  # 计算原始梯度

        # 2. PGD 对抗训练逻辑
        if pgd_attacker is not None:
            pgd_attacker.backup_grad()  # 备份原始梯度

            # 进行 K 次对抗攻击
            for t in range(K):
                # 在 embedding 上添加扰动，is_first_attack 只有第一次为 True
                pgd_attacker.attack(is_first_attack=(t == 0))

                # 如果不是最后一步，需要清空梯度计算新的扰动方向
                if t != K - 1:
                    model.zero_grad()
                else:
                    pgd_attacker.restore_grad()  # 最后一步恢复正常的梯度用于累加

                outputs_adv = model(ids, mask, token_type_ids)
                loss_adv = criterion(outputs_adv, labels)
                loss_adv.backward()  # 反向传播对抗梯度

            pgd_attacker.restore()  # 恢复原始 embedding 参数

        # 3. 参数更新
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    return np.mean(losses)


def eval_fn(model, valid_loader, device):
    model.eval()
    final_targets = []
    final_outputs = []

    with torch.no_grad():
        for data in valid_loader:
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            labels = data['labels'].to(device).view(-1, 1)

            outputs = model(ids, mask, token_type_ids)
            outputs = torch.sigmoid(outputs)  # 转为概率

            final_targets.extend(labels.cpu().detach().numpy().tolist())
            final_outputs.extend(outputs.cpu().detach().numpy().tolist())

    return final_outputs, final_targets


def run():
    conf = Config()
    seed_everything(conf.SEED)

    df_train = pd.read_csv(conf.TRAIN_FILE)

    # K折交叉验证
    skf = StratifiedKFold(n_splits=conf.N_FOLDS, shuffle=True, random_state=conf.SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train['label'])):
        print(f"\n------ Fold {fold + 1}/{conf.N_FOLDS} ------")

        train_df = df_train.iloc[train_idx].reset_index(drop=True)
        valid_df = df_train.iloc[val_idx].reset_index(drop=True)

        train_dataset = FakeNewsDataset(train_df, conf, mode='train')
        valid_dataset = FakeNewsDataset(valid_df, conf, mode='train')

        train_loader = DataLoader(train_dataset, batch_size=conf.BATCH_SIZE, shuffle=True, num_workers=2)
        valid_loader = DataLoader(valid_dataset, batch_size=conf.BATCH_SIZE, shuffle=False, num_workers=2)

        model = FakeNewsModel(conf)
        model.to(conf.DEVICE)

        optimizer = AdamW(model.parameters(), lr=conf.LEARNING_RATE, weight_decay=conf.WEIGHT_DECAY)
        num_train_steps = int(len(train_df) / conf.BATCH_SIZE * conf.EPOCHS)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(num_train_steps * conf.WARMUP_RATIO),
                                                    num_training_steps=num_train_steps)
        criterion = nn.BCEWithLogitsLoss()

        # 初始化对抗训练
        pgd = PGD(model) if conf.USE_PGD else None

        best_auc = 0

        for epoch in range(conf.EPOCHS):
            avg_loss = train_fn(model, train_loader, optimizer, scheduler, conf.DEVICE, criterion, pgd)
            preds, targets = eval_fn(model, valid_loader, conf.DEVICE)
            auc_score = calculate_auc(targets, preds)

            print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | AUC: {auc_score:.4f}")

            if auc_score > best_auc:
                best_auc = auc_score
                torch.save(model.state_dict(), f"{conf.OUTPUT_DIR}/bert_lstm_fold{fold}.pth")
                print(f"--> Best Model Saved! AUC: {best_auc:.4f}")


if __name__ == "__main__":
    run()