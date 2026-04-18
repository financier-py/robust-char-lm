# data_utils.py

import re
import json
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader

from dataset import Vocab, TextAugmenter, RobustDataset
from config import config


def clean_text(text: str) -> str:
    text = text.lower()  # пока что так, так как убрал знаки препинания

    text = re.sub(r"==.*?==", "", text)  # тут убираем заголовки wiki
    text = re.sub(r"\d+", "0", text)  # заменяем все числа на один токен 0

    text = re.sub(r"\.", " . ", text)  # делаем точку отдельным токеном

    text = re.sub(r"[^а-яА-ЯёЁ0\. ]", " ", text)  # все кроме букв, 0 и точки убираем

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_examples(batch):
    chunks = []
    for text in batch["text"]:
        cleaned = clean_text(text)
        words = cleaned.split()

        for i in range(0, len(words), config.max_seq_len):
            chunk = words[i : i + config.max_seq_len]
            if len(chunk) >= 10:
                chunks.append(" ".join(chunk))

    return {"text": chunks}


def prepare_dataloaders(
    limit: int, batch_size: int, val_ratio: float = 0.1, num_workers: int = 2
):
    print("Грузим вики")
    wiki_ds = load_dataset(
        "wikimedia/wikipedia", "20231101.ru", split=f"train[:{limit}]"
    )
    wiki_ds = wiki_ds.select_columns(["text"])

    print("Грузим либрусек книжечки")
    lit_ds_raw = load_dataset(
        "parquet",
        data_files=[
            "hf://datasets/averoo/librusec/data/train-00000-of-00049.parquet",
            "hf://datasets/averoo/librusec/data/train-00001-of-00049.parquet",
        ],
        split="train",
    )

    lit_ds_raw = lit_ds_raw.select(range(min(limit // 8, len(lit_ds_raw))))
    print("размер lit_ds_raw:", len(lit_ds_raw))

    def extract_text(doc_json):
        try:
            doc = json.loads(doc_json)
            res = []
            for body in doc:
                if "body_title" in body:
                    res.append("\n".join(body["body_title"]))
                if "sections" in body:
                    # Итерируемся по массиву секций
                    for section in body["sections"]:
                        if "data" in section and "pars" in section["data"]:
                            # Вытаскиваем длинные параграфы текста
                            res.extend(
                                [
                                    p
                                    for p in section["data"]["pars"]
                                    if isinstance(p, str) and len(p) > 20
                                ]
                            )
            return "\n".join(res)
        except Exception:
            return ""

    print("Парсинг JSON Либрусека...")

    def process_lit_item(example):
        return {"text": extract_text(example["book_json"])}

    lit_ds = lit_ds_raw.map(
        process_lit_item, remove_columns=lit_ds_raw.column_names, num_proc=num_workers
    )

    # убираем пустые и короткие книжки
    lit_ds = lit_ds.filter(lambda x: len(x["text"]) > 100, num_proc=num_workers)

    print("Склеиваем датасеты и перемешиваем...")
    full_ds = concatenate_datasets([wiki_ds, lit_ds])
    full_ds = full_ds.shuffle(seed=67)

    full_ds = full_ds.map(
        chunk_examples,
        batched=True,
        remove_columns=full_ds.column_names,
        num_proc=num_workers,
    )

    split_data = full_ds.train_test_split(test_size=val_ratio, seed=67)
    train_ds = split_data["train"]
    val_ds = split_data["test"]

    print("Делаем словарь")
    chars = sorted(list(config.ALLOWED_CHARS))

    # я задолбался ждать
    sample_size = min(500_000, len(train_ds))
    vocab_data = train_ds.select(range(sample_size))

    def vocab_gen():
        for item in vocab_data:
            yield item["text"]

    vocab = Vocab(chars, vocab_gen())
    augmenter = TextAugmenter(chars=config.ALLOWED_CHARS)

    print("Создаем DataLoader'ы...")
    train_loader = DataLoader(
        RobustDataset(train_ds, vocab, augmenter),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        RobustDataset(val_ds, vocab, augmenter),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )

    print(f"Готово! Train: {len(train_ds)} блоков, Val: {len(val_ds)} блоков")
    print(f"Размер словаря: {len(vocab.word2idx)}")

    return train_loader, val_loader, vocab
