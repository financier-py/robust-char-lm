from dataclasses import dataclass

@dataclass
class Config:
    chars: str = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    max_word_len: int = 20
    max_seq_len: int = 30

    max_words: int = 20_000

    char_emb_dim: int = 16
    cnn_filters: int = 32
    lstm_hidden: int = 64
    lstm_layers: int = 2
    dropout: bool = True


config = Config()