import torch
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from model import RobustLM
from dataset import Vocab
from config import config


@dataclass
class Token:
    original: str
    is_word: bool
    is_title: bool = False
    is_upper: bool = False


class SpellChecker:
    def __init__(self, vocab_path: Path, model_path: Path, device: str = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        with open(vocab_path, "rb") as f:
            self.vocab: Vocab = pickle.load(f)

        self.idx2word = {idx: word for word, idx in self.vocab.word2idx.items()}

        char_vocab_size = len(self.vocab.char2idx)
        word_vocab_size = len(self.vocab.word2idx)

        self.model = RobustLM(char_vocab_size, word_vocab_size)

        weights = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(weights)
        self.model.to(self.device)
        self.model.eval()

    def restore_casing(self, word: str, token: Token) -> str:
        if token.is_upper:
            return word.upper()
        elif token.is_title:
            return word.capitalize()
        return word

    def correct_text(self, text: str) -> str:
        tokens = self.tokenize_text(text)

        words_to_fix = [token.original.lower() for token in tokens if token.is_word]
        if not words_to_fix:
            return text

        encoded_words = []
        for word in words_to_fix:
            word_tensor = self.vocab.encode_word(word)
            encoded_words.append(word_tensor)

        x_tensor = torch.stack(encoded_words).unsqueeze(0)
        x_tensor = x_tensor.to(self.device)

        # типо реальная длина
        length_tens = torch.tensor([x_tensor.size(1)], dtype=torch.long)

        with torch.no_grad():
            logits = self.model(x_tensor, length_tens)
            preds_ids = torch.argmax(logits, dim=-1)[0]

        res = []
        word_ptr = 0  # это указатель

        for token in tokens:
            if not token.is_word:
                res.append(token.original)
            else:
                pred_id = preds_ids[word_ptr].item()

                if pred_id in (0, 1):
                    fixed_word = token.original.lower()
                else:
                    fixed_word = self.idx2word[pred_id]

                final_word = self.restore_casing(fixed_word, token)
                res.append(final_word)

                word_ptr += 1
        return "".join(res)

    @staticmethod
    def tokenize_text(text: str) -> list[Token]:
        parts = re.split(r"([а-яА-ЯёЁ]+)", text)

        tokens = []
        for part in parts:
            if not part:
                continue

            if part.isalpha():
                tokens.append(
                    Token(
                        original=part,
                        is_word=True,
                        is_title=part.istitle(),
                        is_upper=part.isupper(),
                    )
                )
            else:
                tokens.append(Token(original=part, is_word=False))
        return tokens


if __name__ == "__main__":
    vocab_file = Path(__file__).parent / "checkpoints" / "vocab.pkl"
    model_file = Path(__file__).parent / "checkpoints" / "best_model.pt"

    checker = SpellChecker(vocab_path=vocab_file, model_path=model_file)

    print("Введи текст для исправления (или 'выход' / 'exit' для завершения).")
    while True:
        try:
            user_input = input("\nТекст: ").strip()

            if user_input.lower() in ["выход", "exit", "q", "quit"]:
                print("Завершение работы. Пока!")
                break

            if not user_input:
                continue

            corrected = checker.correct_text(user_input)
            print(f"Исправлено: {corrected}")

        except KeyboardInterrupt:
            print("\nЗавершение работы. Пока!")
            break
