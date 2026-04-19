# data_utils.py

import re
import json
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader

from dataset import Vocab, TextAugmenter, RobustDataset
from config import config


WIKI_HEADER_RE = re.compile(r"==.*?==")  # Заголовки wiki
DIGITS_RE = re.compile(r"\d+")  # Любые цифры
DOT_RE = re.compile(r"\.")  # Точки
ALLOWED_CHARS_RE = re.compile(
    r"[^а-яА-ЯёЁ0\. ]"
)  # Всё, кроме кириллицы, 0, точки и пробела
EXTRA_SPACES_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = text.lower()  # пока что так, так как убрал знаки препинания

    text = WIKI_HEADER_RE.sub("", text)
    text = DIGITS_RE.sub("0", text)
    text = DOT_RE.sub(" . ", text)
    text = ALLOWED_CHARS_RE.sub(" ", text)
    text = EXTRA_SPACES_RE.sub(" ", text)

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


def load_wiki_dataset(limit: int):
    wiki_ds = load_dataset(
        "wikimedia/wikipedia", "20231101.ru", split=f"train[:{limit}]"
    )
    wiki_ds = wiki_ds.select_columns(["text"])
    return wiki_ds


def extract_json(doc_json: str) -> str:
    """Парсим JSON книги и достаем оттуда текст"""
    try:
        doc = json.loads(doc_json)
        res = []
        for body in doc:
            if "sections" in body:
                for section in body["sections"]:
                    if "data" in section and "pars" in section["data"]:
                        res.extend(
                            [
                                text
                                for text in section["data"]["pars"]
                                if isinstance(text, str) and len(text) > 20
                            ]
                        )
        return "\n".join(res)
    except (json.JSONDecodeError, TypeError, KeyError):
        return ""


def process_lit_item(example: dict) -> dict:
    return {"text": extract_json(example["book_json"])}


def load_librusec_dataset(limit: int, num_workers: int):
    """Загружает, парсит и фильтрует датасет Либрусека"""
    lit_ds_raw = load_dataset(
        "parquet",
        data_files=[
            "hf://datasets/averoo/librusec/data/train-00000-of-00049.parquet",
            "hf://datasets/averoo/librusec/data/train-00001-of-00049.parquet",
        ],
        split="train",
    )

    lit_ds_raw = lit_ds_raw.select(range(min(limit // 8, len(lit_ds_raw))))
    print("Размер lit_ds_raw:", len(lit_ds_raw))

    lit_ds = lit_ds_raw.map(
        process_lit_item, remove_columns=lit_ds_raw.column_names, num_proc=num_workers
    )
    lit_ds = lit_ds.filter(lambda x: len(x["text"]) > 100, num_proc=num_workers)
    return lit_ds


def prepare_train_val_split(wiki_ds, lit_ds, val_ratio: float, num_workers: int):
    full_ds = concatenate_datasets([wiki_ds, lit_ds])
    full_ds = full_ds.shuffle(seed=config.seed)

    full_ds = full_ds.map(
        chunk_examples,
        batched=True,
        remove_columns=full_ds.column_names,
        num_proc=num_workers,
    )

    split_data = full_ds.train_test_split(test_size=val_ratio, seed=config.seed)
    train_ds = split_data["train"]
    val_ds = split_data["test"]

    return train_ds, val_ds


def build_vocabulary(train_ds):
    chars = sorted(list(config.ALLOWED_CHARS))

    # тут ограничиваем размер датасета, по которому будет собираться словарь, дабы ускорить этот процесс
    sample_size = min(config.limit_ds_vocab, len(train_ds))
    vocab_data = train_ds.select(range(sample_size))

    def vocab_gen():
        for item in vocab_data:
            yield item["text"]

    return Vocab(chars, vocab_gen())


def create_dataloaders(train_ds, val_ds, vocab, batch_size: int, num_workers: int):
    augmenter = TextAugmenter(chars=config.ALLOWED_CHARS)

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

    return train_loader, val_loader


def prepare_dataloaders(
    limit: int, batch_size: int, val_ratio: float = 0.1, num_workers: int = 2
):
    print("Грузим вики")
    wiki_ds = load_wiki_dataset(limit=limit)

    print("Грузим либрусек книжечки")
    lit_ds = load_librusec_dataset(limit=limit, num_workers=num_workers)

    print("Склеиваем датасеты и перемешиваем...")
    train_ds, val_ds = prepare_train_val_split(wiki_ds, lit_ds, val_ratio, num_workers)

    print("Делаем словарь")
    vocab = build_vocabulary(train_ds)

    print("Создаем даталоадеры")
    train_loader, val_loader = create_dataloaders(
        train_ds, val_ds, vocab, batch_size, num_workers
    )

    print(f"Готово! Train: {len(train_ds)} блоков, Val: {len(val_ds)} блоков")
    print(f"Размер словаря: {len(vocab.word2idx)}")

    return train_loader, val_loader, vocab
