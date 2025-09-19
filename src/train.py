import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import sentencepiece as spm
import math
import random

from dataset import TranslationDataset, collate_fn
from model import Encoder, Decoder, Seq2Seq

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Scheduled Sampling Helper
# --------------------------
def get_teacher_forcing_ratio(epoch, max_epochs):
    """
    Scheduled Sampling: start high, decay gradually
    """
    start_ratio = 0.9
    end_ratio = 0.25
    decay = (start_ratio - end_ratio) / max_epochs
    return max(end_ratio, start_ratio - epoch * decay)

# --------------------------
# Train Function
# --------------------------
def train_model():
    # --------------------------
    # Paths
    # --------------------------
    PROCESSED_DIR = os.path.join("data", "processed", "tokenized")
    TRAIN_FILE = os.path.join(PROCESSED_DIR, "train_tok.jsonl")
    VALID_FILE = os.path.join(PROCESSED_DIR, "valid_tok.jsonl")

    vocab_dir = os.path.join("data", "processed", "vocab")
    sp_urdu = spm.SentencePieceProcessor(model_file=os.path.join(vocab_dir, "urdu_bpe.model"))
    sp_roman = spm.SentencePieceProcessor(model_file=os.path.join(vocab_dir, "roman_bpe.model"))
    urdu_vocab_size = sp_urdu.get_piece_size()
    roman_vocab_size = sp_roman.get_piece_size()

    # --------------------------
    # Curriculum: start shorter, expand max_len gradually
    # --------------------------
    max_len_start, max_len_end = 30, 50
    max_epochs = 60

    # --------------------------
    # Datasets & Loaders
    # --------------------------
    train_ds = TranslationDataset(TRAIN_FILE, max_len=max_len_start)
    valid_ds = TranslationDataset(VALID_FILE, max_len=max_len_start)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # --------------------------
    # Model with enhancements
    # --------------------------
    ENC_EMB_DIM = 512
    DEC_EMB_DIM = 512
    HID_DIM = 256   # smaller hidden dim to reduce overfitting

    encoder = Encoder(urdu_vocab_size, ENC_EMB_DIM, HID_DIM, n_layers=2, dropout=0.6)
    decoder = Decoder(roman_vocab_size, DEC_EMB_DIM, HID_DIM, n_layers=4, dropout=0.6, tie_embeddings=True)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=max_epochs,
        steps_per_epoch=max(1, len(train_loader)),
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=100.0
    )

    best_valid_loss = float("inf")
    patience = 12
    min_delta = 0.01
    counter = 0

    # --------------------------
    # Training Loop
    # --------------------------
    for epoch in range(max_epochs):
        # Adjust curriculum max_len gradually
        current_max_len = min(max_len_end, max_len_start + (epoch // 10) * 5)
        train_ds.max_len = current_max_len
        valid_ds.max_len = current_max_len

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

        model.train()
        epoch_loss = 0

        tf_ratio = get_teacher_forcing_ratio(epoch, max_epochs)

        for src, trg in train_loader:
            src, trg = src.to(DEVICE), trg.to(DEVICE)

            optimizer.zero_grad()
            output = model(src, trg, teacher_forcing_ratio=tf_ratio)

            output_dim = output.shape[-1]
            output = output[:,1:].reshape(-1, output_dim)
            trg = trg[:,1:].reshape(-1)

            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)

        # ---------------- Validation ----------------
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, trg in valid_loader:
                src, trg = src.to(DEVICE), trg.to(DEVICE)
                output = model(src, trg, 0)
                output_dim = output.shape[-1]
                output = output[:,1:].reshape(-1, output_dim)
                trg = trg[:,1:].reshape(-1)
                loss = criterion(output, trg)
                val_loss += loss.item()

        val_loss = val_loss / len(valid_loader)

        print(f"Epoch {epoch+1}/{max_epochs} | MaxLen={current_max_len} | TF={tf_ratio:.2f} | "
              f"Train Loss: {train_loss:.4f} | Valid Loss: {val_loss:.4f}")

        # ---------------- Early Stopping ----------------
        if (best_valid_loss - val_loss) > min_delta:
            best_valid_loss = val_loss
            counter = 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_model.pth")
            print("✅ Model improved & saved!")
        else:
            counter += 1
            print(f"⚠️ No improvement. Patience {counter}/{patience}")
            if counter >= patience:
                print("⏹️ Early stopping triggered.")
                break

if __name__ == "__main__":
    train_model()
