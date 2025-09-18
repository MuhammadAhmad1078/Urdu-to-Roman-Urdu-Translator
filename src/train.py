import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from model import Encoder, Decoder, Seq2Seq
import sentencepiece as spm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Dataset Loader
# --------------------------
class TranslationDataset(Dataset):
    def __init__(self, csv_file, sp_urdu, sp_roman, max_len=50):
        self.df = pd.read_csv(csv_file)
        self.sp_urdu = sp_urdu
        self.sp_roman = sp_roman
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        urdu = self.df.iloc[idx]["urdu_text"]
        roman = self.df.iloc[idx]["roman_text"]

        urdu_ids = [self.sp_urdu.bos_id()] + self.sp_urdu.encode(urdu, out_type=int)[:self.max_len-2] + [self.sp_urdu.eos_id()]
        roman_ids = [self.sp_roman.bos_id()] + self.sp_roman.encode(roman, out_type=int)[:self.max_len-2] + [self.sp_roman.eos_id()]

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
    # Base directories (relative)
    DATA_DIR = os.path.join("data", "processed")
    VOCAB_DIR = os.path.join(DATA_DIR, "vocab")
    MODELS_DIR = os.path.join("models")

    # Load tokenizers
    sp_urdu = spm.SentencePieceProcessor(model_file=os.path.join(VOCAB_DIR, "urdu_bpe.model"))
    sp_roman = spm.SentencePieceProcessor(model_file=os.path.join(VOCAB_DIR, "roman_bpe.model"))

    train_ds = TranslationDataset(os.path.join(DATA_DIR, "train.csv"), sp_urdu, sp_roman)
    valid_ds = TranslationDataset(os.path.join(DATA_DIR, "valid.csv"), sp_urdu, sp_roman)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    INPUT_DIM = sp_urdu.get_piece_size()
    OUTPUT_DIM = sp_roman.get_piece_size()
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512

    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, n_layers=2, dropout=0.3)
    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, n_layers=4, dropout=0.3)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore <pad>

    best_valid_loss = float("inf")
    patience = 3   # stop if no improvement for 3 epochs
    counter = 0

    for epoch in range(30):  # allow up to 30 epochs, but early stop
        model.train()
        epoch_loss = 0
        for src, trg in train_loader:
            src, trg = src.to(DEVICE), trg.to(DEVICE)

            optimizer.zero_grad()
            output = model(src, trg)

            # Flatten outputs
            output_dim = output.shape[-1]
            output = output[:,1:].reshape(-1, output_dim)
            trg = trg[:,1:].reshape(-1)

            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)

        # Validation
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

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            counter = 0
            os.makedirs(MODELS_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model.pth"))
            print("✅ Model improved & saved!")
        else:
            counter += 1
            print(f"⚠️ No improvement. Patience {counter}/{patience}")
            if counter >= patience:
                print("⏹️ Early stopping triggered.")
                break

if __name__ == "__main__":
    train_model()
