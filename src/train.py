import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from model import Encoder, Decoder, Seq2Seq
import sentencepiece as spm
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import Levenshtein as Lev

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
# Training Loop with Early Stopping + Metrics
# --------------------------
def train_model():
    # Load tokenizers
    VOCAB_DIR = os.path.join("data", "processed", "vocab")
    sp_urdu = spm.SentencePieceProcessor(model_file=os.path.join(VOCAB_DIR, "urdu_bpe.model"))
    sp_roman = spm.SentencePieceProcessor(model_file=os.path.join(VOCAB_DIR, "roman_bpe.model"))

    train_ds = TranslationDataset("data/processed/train.csv", sp_urdu, sp_roman)
    valid_ds = TranslationDataset("data/processed/valid.csv", sp_urdu, sp_roman)

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

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore <pad>

    best_valid_loss = float("inf")
    patience = 5
    counter = 0

    os.makedirs("results", exist_ok=True)
    log_file = os.path.join("results", "experiment_logs.csv")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,bleu,cer\n")

    for epoch in range(40):  # up to 40 epochs, with early stopping
        # -------------------
        # Training
        # -------------------
        model.train()
        epoch_loss = 0
        for src, trg in train_loader:
            src, trg = src.to(DEVICE), trg.to(DEVICE)

            optimizer.zero_grad()
            teacher_forcing_ratio = max(0.5, 0.8 - epoch * 0.02)
            output = model(src, trg, teacher_forcing_ratio)

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)

        # -------------------
        # Validation
        # -------------------
        model.eval()
        val_loss, bleu_scores, cers = 0, [], []
        with torch.no_grad():
            for src, trg in valid_loader:
                src, trg = src.to(DEVICE), trg.to(DEVICE)

                encoder_outputs, hidden, cell = model.encoder(src)

                batch_size, trg_len = trg.shape
                outputs = torch.zeros(batch_size, trg_len, OUTPUT_DIM).to(DEVICE)
                input = trg[:, 0]  # <sos>

                for t in range(1, trg_len):
                    output, hidden, cell = model.decoder(input, hidden, cell, encoder_outputs)
                    outputs[:, t] = output
                    input = output.argmax(1)

                # Loss
                output_dim = outputs.shape[-1]
                out_flat = outputs[:, 1:].reshape(-1, output_dim)
                trg_flat = trg[:, 1:].reshape(-1)
                loss = criterion(out_flat, trg_flat)
                val_loss += loss.item()

                # Metrics
                preds = outputs.argmax(2).cpu().numpy()
                trgs = trg.cpu().numpy()
                for p, t in zip(preds, trgs):
                    pred_tokens = [sp_roman.id_to_piece(int(i)) for i in p if 0 <= int(i) < sp_roman.get_piece_size()]
                    true_tokens = [sp_roman.id_to_piece(int(i)) for i in t if 0 <= int(i) < sp_roman.get_piece_size()]
                    if true_tokens and pred_tokens:
                        bleu = sentence_bleu([true_tokens], pred_tokens)
                        bleu_scores.append(bleu)
                        cer = Lev.distance(" ".join(true_tokens), " ".join(pred_tokens)) / max(1, len(" ".join(true_tokens)))
                        cers.append(cer)

        val_loss /= len(valid_loader)
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
        avg_cer = np.mean(cers) if cers else 1

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Valid Loss: {val_loss:.4f} | BLEU: {avg_bleu:.4f} | CER: {avg_cer:.4f}")

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch+1},{train_loss:.4f},{val_loss:.4f},{avg_bleu:.4f},{avg_cer:.4f}\n")

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            counter = 0
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
