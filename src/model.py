import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class FakeNewsModel(nn.Module):
    def __init__(self, config):
        super(FakeNewsModel, self).__init__()
        self.bert = BertModel.from_pretrained(config.BERT_PATH)

        # 冻结部分底层参数（可选，显存不足时开启）
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        hidden_size = self.bert.config.hidden_size

        # 增加一个BiLSTM层来捕捉长距离依赖
        self.lstm = nn.LSTM(hidden_size, hidden_size // 2, batch_first=True, bidirectional=True)

        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, ids, mask, token_type_ids):
        # BERT 输出: (batch, seq_len, hidden)
        outputs = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state

        # 经过 LSTM: (batch, seq_len, hidden)
        lstm_out, _ = self.lstm(sequence_output)

        # Attention Pooling
        # weights: (batch, seq_len, 1)
        weights = torch.softmax(self.attention(lstm_out), dim=1)

        # context: (batch, hidden)
        context_vector = torch.sum(weights * lstm_out, dim=1)

        # Final Logits
        logits = self.classifier(context_vector)

        return logits