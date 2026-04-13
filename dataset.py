# dataset.py

import random
import torch
from torch.utils.data import Dataset
from collections import Counter
from typing import Iterable

from config import config


class TextAugmenter:
    def __init__(self, prob: float = 0.4, chars: str = config.ALLOWED_CHARS):
        self.prob = prob
        self.chars = chars
        self.neighbors = config.neigbours
        self.phonetic_rules = config.phonetic_rules

    def swap_chars(self, word: str):
        if len(word) < 2:
            return word
        word_chars = list(word)
        i = random.randint(0, len(word_chars) - 2)
        word_chars[i], word_chars[i + 1] = word_chars[i + 1], word_chars[i]
        return "".join(word_chars)

    def delete_char(self, word: str):
        if len(word) < 3:
            return word
        i = random.randint(0, len(word) - 1)
        return word[:i] + word[i + 1 :]

    def insert_char(self, word: str):
        if len(word) >= config.max_word_len:
            return word
        i = random.randint(0, len(word))
        random_char = random.choice(self.chars.strip())
        return word[:i] + random_char + word[i:]

    def substitute_neighbor(self, word: str):
        if len(word) < 1:
            return word
        i = random.randint(0, len(word) - 1)
        char = word[i].lower()

        if char in self.neighbors:
            new_char = random.choice(self.neighbors[char])
            if word[i].isupper():
                new_char = new_char.upper()
            return word[:i] + new_char + word[i + 1 :]
        return self.substitute_char(word)

    def substitute_char(self, word: str):
        if len(word) < 1:
            return word
        i = random.randint(0, len(word) - 1)
        j_char = random.choice(self.chars)
        if word[i].isupper():
            j_char = j_char.upper()
        else:
            j_char = j_char.lower()
        return word[:i] + j_char + word[i + 1 :]
    
    def apply_phonetic_error(self, word: str):
        if len(word) < 2:
            return word

        possible_replacements = []
        
        for target, replacements in self.phonetic_rules.items():
            if target in word:
                for rep in replacements:
                    possible_replacements.append((target, rep))

        if not possible_replacements:
            return word

        target, replacement = random.choice(possible_replacements)

        return word.replace(target, replacement, 1)

    def apply_noise(self, word: str):
        cur_prob = self.prob * (min(len(word), 10) / 5)

        if random.random() > cur_prob:
            return word

        strategies = [
            self.swap_chars,
            self.delete_char,
            self.insert_char,
            self.substitute_char,
            self.substitute_neighbor,
            self.apply_phonetic_error,
            self.apply_phonetic_error
        ]

        strategy = random.choice(strategies)
        return strategy(word)


class Vocab:
    def __init__(self, char_list: list[str], texts_iterator: Iterable[str]):
        self.char2idx = {char: i + 2 for i, char in enumerate(char_list)}
        self.char2idx["<PAD>"] = 0
        self.char2idx["<UNK>"] = 1

        self.word2idx = self._build_vocab(texts_iterator)

    def encode_word(self, word: str) -> torch.Tensor:
        indices = [self.char2idx.get(char, 1) for char in word]
        indices = indices[: config.max_word_len]

        pad_len = config.max_word_len - len(indices)
        if pad_len > 0:
            indices += [0] * pad_len

        return torch.tensor(indices)

    def _build_vocab(self, texts_iterator: Iterable[str]):
        cnt_words = Counter()

        for text in texts_iterator:
            cnt_words.update(text.split())

        most_common = cnt_words.most_common(config.max_words)

        vocab = {"<PAD>": 0, "<UNK>": 1}
        for i, (word, _) in enumerate(most_common):
            vocab[word] = i + 2
        return vocab


class RobustDataset(Dataset):
    def __init__(self, hf_dataset, vocab: Vocab, augmenter: TextAugmenter):
        self.hf_dataset = hf_dataset
        self.vocab = vocab
        self.augmenter = augmenter

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx: int):
        sentence = self.hf_dataset[idx]["text"]
        words = sentence.split()[: config.max_seq_len]

        dirty_words_tensor = []
        clean_words_ids = []
        is_noisy = []

        for word in words:
            noisy_word = self.augmenter.apply_noise(word)
            dirty_words_tensor.append(self.vocab.encode_word(noisy_word))

            is_noisy.append(1 if noisy_word != word else 0)

            clean_words_ids.append(self.vocab.word2idx.get(word, 1))

        pad_len = config.max_seq_len - len(dirty_words_tensor)

        if pad_len > 0:
            for _ in range(pad_len):
                dirty_words_tensor.append(
                    torch.zeros(config.max_word_len, dtype=torch.long)
                )
                clean_words_ids.append(0)
                is_noisy.append(0)

        return {
            "x": torch.stack(dirty_words_tensor),
            "y": torch.tensor(clean_words_ids, dtype=torch.long),
            "is_noisy": torch.tensor(is_noisy, dtype=torch.bool),
        }
