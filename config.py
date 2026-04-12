from dataclasses import dataclass

@dataclass
class Config:
    chars: str = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    max_word_len: int = 20
    max_seq_len: int = 30

    max_words: int = 20_000


config = Config()