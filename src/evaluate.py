import os
import ast
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from model import Encoder, Decoder, Seq2Seq
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import Levenshtein as Lev
import sentencepiece as spm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Dataset Loader
# --------------------------
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, max_len=50):
        self.df = pd.read_csv(csv_file)
        # Filter out malformed rows (e.g., merge conflict markers)
        def is_valid(cell):
            if not isinstance(cell, str):
                return False
            s = cell.strip()
            if any(mark in s for mark in [">>>>>>>", "<<<<<<<", "======="]):
                return False
            return s.startswith("[") and s.endswith("]")

        self.df = self.df[self.df["urdu_ids"].apply(is_valid) & self.df["roman_ids"].apply(is_valid)].reset_index(drop=True)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        urdu_ids = ast.literal_eval(self.df.iloc[idx]["urdu_ids"])
        roman_ids = ast.literal_eval(self.df.iloc[idx]["roman_ids"])

        urdu_ids = urdu_ids[:self.max_len]
        roman_ids = roman_ids[:self.max_len]

        return torch.tensor(urdu_ids), torch.tensor(roman_ids)

def collate_fn(batch):
    src, trg = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    trg = nn.utils.rnn.pad_sequence(trg, batch_first=True, padding_value=0)
    return src, trg

# --------------------------
# Evaluation
# --------------------------
def evaluate_model():
    TEST_FILE = os.path.join("data", "processed", "tokenized", "test_tok.csv")
    test_ds = TranslationDataset(TEST_FILE)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # vocab sizes from SentencePiece models
    vocab_dir = os.path.join("data", "processed", "vocab")
    sp_urdu = spm.SentencePieceProcessor(model_file=os.path.join(vocab_dir, "urdu_bpe.model"))
    sp_roman = spm.SentencePieceProcessor(model_file=os.path.join(vocab_dir, "roman_bpe.model"))
    urdu_vocab_size = sp_urdu.get_piece_size()
    roman_vocab_size = sp_roman.get_piece_size()

    encoder = Encoder(urdu_vocab_size, 512, 512, n_layers=2, dropout=0.5)
    decoder = Decoder(roman_vocab_size, 512, 512, n_layers=4, dropout=0.5)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    bleu_scores, cers, losses = [], [], []
    smoothie = SmoothingFunction().method4

    with torch.no_grad():
        for src, trg in test_loader:
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            output = model(src, trg, 0)  # no teacher forcing

            # compute loss
            output_dim = output.shape[-1]
            out_flat = output[:,1:].reshape(-1, output_dim)
            trg_flat = trg[:,1:].reshape(-1)
            loss = criterion(out_flat, trg_flat)
            losses.append(loss.item())

            # predictions
            # Simple no-repeat trigram blocking at evaluation time
            logits_full = output.clone()  # [B, T, V]
            B, T, V = logits_full.shape
            preds = torch.zeros(B, T, dtype=torch.long)
            for b in range(B):
                seen = {}
                for t in range(T):
                    logits_t = logits_full[b, t]
                    if t >= 3:
                        key = (int(preds[b, t-2].item()), int(preds[b, t-1].item()))
                        ban = seen.get(key)
                        if ban is not None:
                            logits_t[ban] = -1e9
                    pred_t = int(torch.argmax(logits_t).item())
                    preds[b, t] = pred_t
                    if t >= 2:
                        key2 = (int(preds[b, t-1].item()), int(pred_t))
                        seen[key2] = pred_t
            preds = preds.cpu().numpy()
            trgs = trg.cpu().numpy()

            for p, t in zip(preds, trgs):
                # Trim at EOS if present
                def trim_at_eos(seq, eos_id):
                    out = []
                    for tok in seq:
                        if tok == eos_id:
                            break
                        if tok != 0:
                            out.append(int(tok))
                    return out

                p_trim = trim_at_eos(p[1:], sp_roman.eos_id())  # skip BOS position
                t_trim = trim_at_eos(t[1:], sp_roman.eos_id())

                if t_trim and p_trim:
                    pred_text = sp_roman.decode(p_trim)
                    true_text = sp_roman.decode(t_trim)

                    # BLEU on word tokens from decoded text
                    pred_words = pred_text.split()
                    true_words = true_text.split()

                    bleu = sentence_bleu([true_words], pred_words, smoothing_function=smoothie)
                    bleu_scores.append(bleu)

                    cer = Lev.distance(true_text, pred_text) / max(1, len(true_text))
                    cers.append(cer)

    avg_loss = np.mean(losses)
    perplexity = np.exp(avg_loss)
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    avg_cer = np.mean(cers) if cers else 1.0

    print(f"Test Results:")
    print(f"- Loss: {avg_loss:.4f}")
    print(f"- Perplexity: {perplexity:.4f}")
    print(f"- BLEU: {avg_bleu:.4f}")
    print(f"- CER: {avg_cer:.4f}")

    # show some qualitative samples
    print("\n🔹 Sample Predictions:")
    for i in range(5):
        urdu_ids = ast.literal_eval(test_ds.df.iloc[i]["urdu_ids"])
        roman_true_ids = ast.literal_eval(test_ds.df.iloc[i]["roman_ids"])

        src = torch.tensor([urdu_ids]).to(DEVICE)
        trg = torch.tensor([roman_true_ids]).to(DEVICE)

        output = model(src, trg, 0)
        pred_ids = output.argmax(2).cpu().numpy()[0]

        # Trim and decode for readability
        def trim(seq, eos):
            out = []
            for tok in seq[1:]:  # skip BOS
                if tok == eos:
                    break
                if tok != 0:
                    out.append(int(tok))
            return out

        roman_pred = sp_roman.decode(trim(pred_ids, sp_roman.eos_id()))
        roman_true = sp_roman.decode(trim(roman_true_ids, sp_roman.eos_id()))

        print(f"\nUrdu IDs: {urdu_ids}")
        print(f"Target Roman: {roman_true}")
        print(f"Predicted Roman: {roman_pred}")


if __name__ == "__main__":
    evaluate_model()
