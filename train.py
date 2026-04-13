# train.py

import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

from config import config
from data_utils import prepare_dataloaders
from model import RobustLM


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # исключительно только для wandb
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('__')}
    
    wandb.init(
        project="robust-lm-spelling", 
        name="char-cnn-lstm-run1",
        config=config_dict
    )

    print("готовится датасет")
    train_loader, val_loader, vocab = prepare_dataloaders(
        limit=config.wiki_limit,
        batch_size=config.batch_size,
        val_ratio=0.1,
        num_workers=8,
    )

    # ====== Тут сохраняем словарь ======
    os.makedirs("checkpoints", exist_ok=True)
    vocab_path = "checkpoints/vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    wandb.save(vocab_path, base_path=".")
    print(f'Словарь сохранен в {vocab_path}')

    char_vocab_size = len(vocab.char2idx)
    word_vocab_size = len(vocab.word2idx)
    print(f"Кол-во слов {word_vocab_size}")

    model = RobustLM(char_vocab_size, word_vocab_size).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs} [Train]")

        for batch in train_pbar:
            x = batch["x"].to(device)  # [Batch, SeqLen, WordLen]
            y = batch["y"].to(device)  # [Batch, SeqLen]

            optimizer.zero_grad()
            logits = model(x)  # [Batch, SeqLen, VocabSize]

            # [Batch, Seq, VocabSize] -> [Batch * Seq, VocabSize]
            logits_flat = logits.view(-1, word_vocab_size)
            # [Batch, Seq] -> [Batch * Seq]
            y_flat = y.view(-1)

            loss = criterion(logits_flat, y_flat)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.grad_norm
            )

            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0

        val_pbar = tqdm(
            val_loader, desc=f"Epoch {epoch}/{config.epochs} [Val]", leave=False
        )

        with torch.no_grad():
            for batch in val_pbar:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                is_noisy = batch["is_noisy"].to(device)

                logits = model(x)
                logits_flat = logits.view(-1, word_vocab_size)
                y_flat = y.view(-1)
                is_noisy_flat = is_noisy.view(-1)

                loss = criterion(logits_flat, y_flat)
                val_loss += loss.item()

                preds = torch.argmax(logits_flat, dim=1)

                mask = (y_flat != 0) & (y_flat != 1) & is_noisy_flat
                if mask.sum() > 0:
                    correct_preds += (preds[mask] == y_flat[mask]).sum ().item()
                    total_preds += mask.sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_preds / total_preds if total_preds > 0 else 0

        scheduler.step(avg_val_loss)

        print(
            f"Эпоха {epoch} Итог: "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "lr": optimizer.param_groups[0]['lr']
        })

        # ====== Тут сохраняем веса ======
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            weights_path = 'checkpoints/best_model.pt'
            torch.save(model.state_dict(), weights_path)
            wandb.save(weights_path, base_path=".")
    wandb.finish()


if __name__ == "__main__":
    train()
