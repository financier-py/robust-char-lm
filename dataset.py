import random
import torch
from torch.utils.data import Dataset
from collections import Counter

from config import config


class TextAugmenter:
    def __init__(self, prob: float = 0.1, chars: str = "абв"):
        self.prob = prob
        self.chars = chars

    def swap_chars(self, word: str):
        if len(word) < 4:
            return word

        word_chars = list(word)
        i = random.randint(0, len(word_chars) - 2)
        word_chars[i], word_chars[i + 1] = word_chars[i + 1], word_chars[i]
        return "".join(word_chars)

    def substitute_char(self, word: str):
        if len(word) < 3:
            return word

        i_word = random.randint(0, len(word) - 1)
        j_char = random.choice(self.chars)

        if word[i_word].isupper():
            j_char = j_char.upper()
        else:
            j_char = j_char.lower()

        return word[:i_word] + j_char + word[i_word + 1 :]

    def delete_char(self, word: str):
        if len(word) < 4:
            return word

        i_word = random.randint(0, len(word) - 1)
        return word[:i_word] + word[i_word + 1 :]

    def apply_noise(self, word):
        cur_prob = self.prob * (len(word) / 5)
        if random.random() > cur_prob:
            return word

        strategy = random.choice(
            [self.swap_chars, self.substitute_char, self.delete_char]
        )
        return strategy(word)


class Vocab:
    def __init__(self, char_list: list[str], all_texts: list[str]):
        self.char2idx = {char: i + 2 for i, char in enumerate(char_list)}
        self.char2idx["<PAD>"] = 0
        self.char2idx["<UNK>"] = 1

        self.word2idx = self._build_vocab(all_texts)

    def encode_word(self, word: str) -> torch.Tensor:
        indices = [self.char2idx.get(char, 1) for char in word]

        indices = indices[: config.max_word_len]

        pad_len = config.max_word_len - len(indices)
        if pad_len > 0:
            indices += [0] * pad_len

        return torch.tensor(indices)

    def _build_vocab(self, texts: list[str]):
        all_words = []
        for text in texts:
            all_words.extend(text.split())
        cnt_words = Counter(all_words).most_common(config.max_words)

        vocab = {"<PAD>": 0, "<UNK>": 1}
        for i, (word, _) in enumerate(cnt_words):
            vocab[word] = i + 2
        return vocab


class RobustDataset(Dataset):
    def __init__(self, texts: list[str], vocab: Vocab, augmenter: TextAugmenter):
        self.texts = texts
        self.vocab = vocab
        self.augmenter = augmenter

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        sentence = self.texts[idx]
        words = sentence.split()[: config.max_seq_len]

        dirty_words_tensor = []
        clean_words_ids = []

        for word in words:
            noisy_word = self.augmenter.apply_noise(word)
            dirty_words_tensor.append(self.vocab.encode_word(noisy_word))
            clean_words_ids.append(self.vocab.word2idx.get(word, 1))

        pad_len = config.max_seq_len - len(dirty_words_tensor)

        if pad_len > 0:
            pad_word = torch.zeros(config.max_word_len, dtype=torch.long)
            dirty_words_tensor += [pad_word] * pad_len
            clean_words_ids += [0] * pad_len  # '<PAD>' -> 0

        return {
            "x": torch.stack(dirty_words_tensor),
            "y": torch.tensor(clean_words_ids, dtype=torch.long),
        }
