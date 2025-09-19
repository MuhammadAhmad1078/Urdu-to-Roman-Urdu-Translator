import os
import ast
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from model import Encoder, Decoder, Seq2Seq
import sentencepiece as spm
from torch.optim.lr_scheduler import ReduceLROnPlateau

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Dataset Loader (use tokenized CSVs directly)
# --------------------------
class TranslationDataset(Dataset):
    def __init__(self, csv_file, max_len=50):
        self.df = pd.read_csv(csv_file)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Parse tokenized IDs from string to list of ints
        urdu_ids = ast.literal_eval(self.df.iloc[idx]["urdu_ids"])
        roman_ids = ast.literal_eval(self.df.iloc[idx]["roman_ids"])

        # Truncate if too long
        urdu_ids = urdu_ids[:self.max_len]
        roman_ids = roman_ids[:self.max_len]

        return torch.tensor(urdu_ids), torch.tensor(roman_ids)


def collate_fn(batch):
    src, trg = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    trg = nn.utils.rnn.pad_sequence(trg, batch_first=True, padding_value=0)
    return src, trg

# --------------------------
# Training Loop with Early Stopping
# --------------------------
def train_model():
    # Relative paths
    PROCESSED_DIR = os.path.join("data", "processed", "tokenized")
    TRAIN_FILE = os.path.join(PROCESSED_DIR, "train_tok.csv")
    VALID_FILE = os.path.join(PROCESSED_DIR, "valid_tok.csv")

    # Load datasets
    train_ds = TranslationDataset(TRAIN_FILE)
    valid_ds = TranslationDataset(VALID_FILE)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # Define model dims from SentencePiece vocab sizes to include special tokens
    vocab_dir = os.path.join("data", "processed", "vocab")
    sp_urdu = spm.SentencePieceProcessor(model_file=os.path.join(vocab_dir, "urdu_bpe.model"))
    sp_roman = spm.SentencePieceProcessor(model_file=os.path.join(vocab_dir, "roman_bpe.model"))
    urdu_vocab_size = sp_urdu.get_piece_size()
    roman_vocab_size = sp_roman.get_piece_size()

    ENC_EMB_DIM = 512
    DEC_EMB_DIM = 512
    HID_DIM = 512

    encoder = Encoder(urdu_vocab_size, ENC_EMB_DIM, HID_DIM, n_layers=2, dropout=0.5)
    decoder = Decoder(roman_vocab_size, DEC_EMB_DIM, HID_DIM, n_layers=4, dropout=0.5)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    # Label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_valid_loss = float("inf")
    patience = 7
    counter = 0
    # Adaptive controls
    base_dropout = 0.3
    max_dropout = 0.5
    current_dropout = base_dropout

    max_epochs = 60
    for epoch in range(max_epochs):
        # ---------------- Train ----------------
        model.train()
        epoch_loss = 0
        # Decay teacher forcing from 0.6 → 0.2 over training
        tf_ratio = max(0.2, 0.6 * (1.0 - epoch / max_epochs))
        # If on plateau, temporarily boost TF to stabilize
        if counter > 0:
            tf_ratio = max(tf_ratio, 0.5)
        for src, trg in train_loader:
            src, trg = src.to(DEVICE), trg.to(DEVICE)

            optimizer.zero_grad()
            output = model(src, trg, teacher_forcing_ratio=tf_ratio)

            # Flatten
            output_dim = output.shape[-1]
            output = output[:,1:].reshape(-1, output_dim)
            trg = trg[:,1:].reshape(-1)

            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)

        # ---------------- Validation ----------------
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, trg in valid_loader:
                src, trg = src.to(DEVICE), trg.to(DEVICE)
                output = model(src, trg, 0)  # no teacher forcing
                output_dim = output.shape[-1]
                output = output[:,1:].reshape(-1, output_dim)
                trg = trg[:,1:].reshape(-1)
                loss = criterion(output, trg)
                val_loss += loss.item()

        val_loss = val_loss / len(valid_loader)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Valid Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        # ---------------- Early Stopping & Adaptation ----------------
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            counter = 0
            # Reset adaptive dropout on improvement
            current_dropout = base_dropout
            model.encoder.dropout.p = current_dropout
            model.decoder.dropout.p = current_dropout
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_model.pth")
            print("✅ Model improved & saved!")
        else:
            counter += 1
            print(f"⚠️ No improvement. Patience {counter}/{patience}")
            # Increase regularization slightly on plateau
            if current_dropout < max_dropout:
                current_dropout = min(max_dropout, current_dropout + 0.05)
                model.encoder.dropout.p = current_dropout
                model.decoder.dropout.p = current_dropout
                print(f"↘️ Increasing dropout to {current_dropout:.2f} to regularize.")
            if counter >= patience:
                print("⏹️ Early stopping triggered.")
                break


if __name__ == "__main__":
    train_model()
