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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Dataset Loader
# --------------------------
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, max_len=50):
        self.df = pd.read_csv(csv_file)
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

    # infer vocab size from dataset
    urdu_vocab_size = max(max(ast.literal_eval(ids)) for ids in test_ds.df["urdu_ids"]) + 1
    roman_vocab_size = max(max(ast.literal_eval(ids)) for ids in test_ds.df["roman_ids"]) + 1

    encoder = Encoder(urdu_vocab_size, 256, 512, n_layers=2, dropout=0.3)
    decoder = Decoder(roman_vocab_size, 256, 512, n_layers=4, dropout=0.3)
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
            preds = output.argmax(2).cpu().numpy()
            trgs = trg.cpu().numpy()

            for p, t in zip(preds, trgs):
                pred_tokens = [str(i) for i in p if i != 0]  # skip padding
                true_tokens = [str(i) for i in t if i != 0]

                if true_tokens and pred_tokens:
                    bleu = sentence_bleu([true_tokens], pred_tokens, smoothing_function=smoothie)
                    bleu_scores.append(bleu)

                    cer = Lev.distance(" ".join(true_tokens), " ".join(pred_tokens)) / max(1, len(" ".join(true_tokens)))
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

        roman_pred = " ".join([str(i) for i in pred_ids if i != 0])
        roman_true = " ".join([str(i) for i in roman_true_ids if i != 0])

        print(f"\nUrdu IDs: {urdu_ids}")
        print(f"Target Roman IDs: {roman_true}")
        print(f"Predicted Roman IDs: {roman_pred}")


if __name__ == "__main__":
    evaluate_model()
