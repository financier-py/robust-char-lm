import torch
import torch.nn as nn

import torch.nn.functional as F

from config import config


class CharCNN(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()

        # Embedding: каждый символ -> вектор
        # входит что-то типо [*, ] (индексы)
        # выходит уже [*, char_emb_dim]
        self.char_emb = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=config.char_emb_dim, padding_idx=0
        )

        # conv1d в питорче ожидает [batch, channels_in, length]
        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=config.char_emb_dim,  # те самые channels_in
                    out_channels=config.cnn_filters,  # channels out
                    kernel_size=k,
                )
                for k in [2, 3, 4, 5]
            ]
        )
        # на выходе как раз получим [batch, channels_out, length - k + 1]

        # у меня четыре фильтра
        tot_filters = config.cnn_filters * 4
        # проецируем в LSTM hidden
        self.projector = nn.Linear(
            in_features=tot_filters, out_features=config.lstm_hidden
        )

    def forward(self, x: torch.Tensor):
        # x: [батч, длина_предл, длина_слова]
        batch_size, seq_len, word_len = x.shape

        # -> [B*S, W]
        # схлопываем батчи, теперь у нас просто список слов
        flat_x = x.reshape(batch_size * seq_len, word_len)

        # -> [B*S, W, char_emb_dim]
        char_embeds = self.char_emb(flat_x)
        # опять же channels нужны в середине
        # чтобы channels были посередине, т.е. [B*S, word_len, char_dim] -> [B*S, char_dim, word_len]
        char_embeds = char_embeds.permute(0, 2, 1)

        conv_outputs = []
        for conv in self.convolutions:
            # -> [B*S, cnn_filters, W - k + 1]
            features = conv(char_embeds)
            features = F.relu(features)

            # max пулинг по длине слова
            # -> [B*S, cnn_filters]
            # по сути берем самый "яркий" признак
            pooled, _ = torch.max(features, dim=2)
            conv_outputs.append(pooled)

        # объединяем все признаки для слова
        # [B*S, cnn_filters * 4]
        word_features = torch.cat(conv_outputs, dim=1)
        # [B*S, lstm_hidden]
        word_embeds = self.projector(word_features)
        # [B, S, lstm_hidden]
        return word_embeds.reshape(batch_size, seq_len, -1)


class HighWay(nn.Module):
    def __init__(self, size: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers

        # just проекция
        self.proj = nn.ModuleList([nn.Linear(size, size) for k in range(num_layers)])

        # гейт решает, что запомнить, что отбросить
        self.gate = nn.ModuleList([nn.Linear(size, size) for k in range(num_layers)])

    def forward(self, x: torch.Tensor):
        for k in range(self.num_layers):
            new_x = F.relu(self.proj(x))
            gate_w = torch.sigmoid(self.gate(x))

            x = (gate_w * new_x) + ((1 - gate_w) * x)
        return x


class RobustLm(nn.Module):
    def __init__(self, char_vocab_size: int, word_vocab_size: int):
        super().__init__()

        self.char_cnn = CharCNN(char_vocab_size)
        self.highway = HighWay(config.lstm_hidden)

        self.lstm = nn.LSTM(
            input_size=config.lstm_hidden,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0,
        )

        self.classificator = nn.Linear(config.lstm_hidden, word_vocab_size)

    def forward(self, x: torch.Tensor):
        embed = self.char_cnn(x)
        embed = self.highway(embed)

        lstm_out, _ = self.lstm(embed)

        logits = self.classificator(lstm_out)
        return logits
