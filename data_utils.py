# data_utils.py

import re
from datasets import load_dataset
from torch.utils.data import DataLoader

from dataset import Vocab, TextAugmenter, RobustDataset
from config import config


def clean_text(text: str) -> str:
    text = text.lower() # пока что так, так как убрал знаки препинания
 
    text = re.sub(r"==.*?==", "", text) # тут убираем заголовки wiki
    text = re.sub(r'\d+', '0', text) # заменяем все числа на один токен 0

    text = re.sub(r'\.', ' . ', text) # делаем точку отдельным токеном
    
    text = re.sub(r'[^а-яА-ЯёЁ0\. ]', " ", text) # все кроме букв, 0 и точки убираем
    
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_dataloaders(
    limit: int, batch_size: int, val_ratio: float = 0.1, num_workers: int = 2
):
    full_ds = load_dataset(
        "wikimedia/wikipedia", "20231101.ru", split=f"train[:{limit}]"
    )

    full_ds = full_ds.filter(
        lambda x: len(x["text"]) > 300, num_proc=num_workers
    )
    full_ds = full_ds.map(
        lambda batch: {"text": [clean_text(t) for t in batch["text"]]},
        batched=True,
        num_proc=num_workers,
    )

    split_data = full_ds.train_test_split(test_size=val_ratio, seed=67)
    train_ds = split_data["train"]
    val_ds = split_data["test"]

    def train_text_generator():
        for item in train_ds:
            yield item["text"]  # type: ignore

    chars = sorted(list(config.ALLOWED_CHARS))
    vocab = Vocab(chars, train_text_generator())

    augmenter = TextAugmenter(chars=config.ALLOWED_CHARS)

    my_train_ds = RobustDataset(train_ds, vocab, augmenter)
    my_val_ds = RobustDataset(val_ds, vocab, augmenter)

    train_loader = DataLoader(
        my_train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        my_val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True
    )

    print(f"Dataset split: Train={len(train_ds)} docs, Val={len(val_ds)} docs")
    print(f"Vocab size: {len(vocab.word2idx)} words")

    return train_loader, val_loader, vocab