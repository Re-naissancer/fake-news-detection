import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from .utils import clean_text


class FakeNewsDataset(Dataset):
    def __init__(self, df, config, mode='train'):
        self.df = df
        self.config = config
        self.mode = mode
        self.tokenizer = BertTokenizer.from_pretrained(config.BERT_PATH)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = clean_text(row['text'])

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,  # 加[CLS]和[SEP]token
            max_length=self.config.MAX_LEN,
            padding='max_length',  # 若文本过短，补齐至最大长度
            truncation=True,  # 若文本过长，截断至最大长度
            return_token_type_ids=True
        )

        ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long)

        if self.mode == 'train':
            label = torch.tensor(row['label'], dtype=torch.float)
            return {
                'ids': ids,
                'mask': mask,
                'token_type_ids': token_type_ids,
                'labels': label
            }
        else:
            return {
                'ids': ids,
                'mask': mask,
                'token_type_ids': token_type_ids
            }