import os, random, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import sentencepiece as spm
import pandas as pd

from dataset import TranslationDataset, collate_fn
from model import Encoder, Decoder, Seq2Seq

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1337
random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def get_teacher_forcing_ratio(epoch, max_epochs):
    start, end = 0.9, 0.25
    tf = max(end, start - (start - end) * (epoch / max_epochs))
    if epoch > 20:
        tf += 0.05
        tf = min(tf, 0.95)
    return tf

def train_model():
    TOK_DIR = os.path.join("data", "processed", "tokenized")
    TRAIN_FILE = os.path.join(TOK_DIR, "train_tok.jsonl")
    VALID_FILE = os.path.join(TOK_DIR, "valid_tok.jsonl")

    vocab_dir = os.path.join("data", "processed", "vocab")
    sp_urdu = spm.SentencePieceProcessor(model_file=os.path.join(vocab_dir, "urdu_bpe.model"))
    sp_roman = spm.SentencePieceProcessor(model_file=os.path.join(vocab_dir, "roman_bpe.model"))
    urdu_vocab_size = sp_urdu.get_piece_size()
    roman_vocab_size = sp_roman.get_piece_size()

    max_len_start, max_len_end = 30, 50
    max_epochs = 60

    train_ds = TranslationDataset(TRAIN_FILE, max_len=max_len_start)
    valid_ds = TranslationDataset(VALID_FILE, max_len=max_len_start)

    def make_loaders():
        return (
            DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn),
            DataLoader(valid_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
        )
    train_loader, valid_loader = make_loaders()

    ENC_EMB_DIM, DEC_EMB_DIM, HID_DIM = 512, 512, 256
    encoder = Encoder(urdu_vocab_size, ENC_EMB_DIM, HID_DIM, n_layers=3, dropout=0.6)
    decoder = Decoder(roman_vocab_size, DEC_EMB_DIM, HID_DIM, n_layers=3, dropout=0.7, tie_embeddings=True)
    decoder.sos_idx, decoder.eos_idx = sp_roman.bos_id(), sp_roman.eos_id()
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.05)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, epochs=max_epochs,
                           steps_per_epoch=max(1, len(train_loader)), pct_start=0.1,
                           div_factor=10.0, final_div_factor=100.0)

    best_valid, patience, counter = float("inf"), 16, 0
    history = []

    for epoch in range(max_epochs):
        curr_max_len = min(max_len_end, max_len_start + (epoch // 10) * 5)
        train_ds.max_len, valid_ds.max_len = curr_max_len, curr_max_len
        train_loader, valid_loader = make_loaders()

        model.train()
        tf_ratio, tr_loss = get_teacher_forcing_ratio(epoch, max_epochs), 0.0
        for src, trg, src_len, _ in train_loader:
            src, trg, src_len = src.to(DEVICE), trg.to(DEVICE), src_len.to(DEVICE)
            optimizer.zero_grad()
            out = model(src, trg, teacher_forcing_ratio=tf_ratio, lengths=src_len)
            V = out.shape[-1]
            loss = criterion(out[:, 1:].reshape(-1, V), trg[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for src, trg, src_len, _ in valid_loader:
                src, trg, src_len = src.to(DEVICE), trg.to(DEVICE), src_len.to(DEVICE)
                out = model(src, trg, teacher_forcing_ratio=0.0, lengths=src_len)
                V = out.shape[-1]
                loss = criterion(out[:, 1:].reshape(-1, V), trg[:, 1:].reshape(-1))
                va_loss += loss.item()
        va_loss /= len(valid_loader)

        print(f"Epoch {epoch+1}/{max_epochs} | MaxLen={curr_max_len} | TF={tf_ratio:.2f} | "
              f"Train: {tr_loss:.4f} | Valid: {va_loss:.4f}")
        history.append({"epoch": epoch+1, "maxlen": curr_max_len, "tf": tf_ratio,
                        "train_loss": tr_loss, "valid_loss": va_loss})

        if va_loss + 1e-6 < best_valid:
            best_valid, counter = va_loss, 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_model.pth")
            print("✅ Model improved & saved!")
        else:
            counter += 1
            print(f"⚠️ No improvement. Patience {counter}/{patience}")
            if counter >= patience:
                print("⏹️ Early stopping triggered."); break

    pd.DataFrame(history).to_csv("models/history.csv", index=False, encoding="utf-8")

if __name__ == "__main__":
    train_model()
