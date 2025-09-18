import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from model import Encoder, Decoder, Seq2Seq
import sentencepiece as spm
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import Levenshtein as Lev

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Dataset Loader (reuse from train.py)
# --------------------------
class TranslationDataset(torch.utils.data.Dataset):
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
# Evaluation
# --------------------------
def evaluate_model():
    sp_urdu = spm.SentencePieceProcessor(model_file="data/processed/vocab/urdu_bpe.model")
    sp_roman = spm.SentencePieceProcessor(model_file="data/processed/vocab/roman_bpe.model")

    test_ds = TranslationDataset("data/processed/test.csv", sp_urdu, sp_roman)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    INPUT_DIM = sp_urdu.get_piece_size()
    OUTPUT_DIM = sp_roman.get_piece_size()

    encoder = Encoder(INPUT_DIM, 256, 512, n_layers=2, dropout=0.3)
    decoder = Decoder(OUTPUT_DIM, 256, 512, n_layers=4, dropout=0.3)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
    model.eval()

    bleu_scores, cers = [], []
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    losses = []

    with torch.no_grad():
        for src, trg in test_loader:
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            output = model(src, trg, 0)  # no teacher forcing

            # Loss
            output_dim = output.shape[-1]
            out_flat = output[:,1:].reshape(-1, output_dim)
            trg_flat = trg[:,1:].reshape(-1)
            loss = criterion(out_flat, trg_flat)
            losses.append(loss.item())

            # Metrics
            preds = output.argmax(2).cpu().numpy()
            trgs = trg.cpu().numpy()

            for p, t in zip(preds, trgs):
                pred_tokens = [sp_roman.id_to_piece(id) for id in p if id not in [0, sp_roman.pad_id()]]
                true_tokens = [sp_roman.id_to_piece(id) for id in t if id not in [0, sp_roman.pad_id()]]

                if true_tokens and pred_tokens:
                    bleu = sentence_bleu([true_tokens], pred_tokens)
                    bleu_scores.append(bleu)
                    cer = Lev.distance(" ".join(true_tokens), " ".join(pred_tokens)) / max(1, len(" ".join(true_tokens)))
                    cers.append(cer)

    avg_loss = np.mean(losses)
    perplexity = np.exp(avg_loss)
    avg_bleu = np.mean(bleu_scores)
    avg_cer = np.mean(cers)

    print(f"Test Results:")
    print(f"- Loss: {avg_loss:.4f}")
    print(f"- Perplexity: {perplexity:.4f}")
    print(f"- BLEU: {avg_bleu:.4f}")
    print(f"- CER: {avg_cer:.4f}")

    # Show some qualitative examples
    print("\n🔹 Sample Translations:")
    for i in range(5):
        urdu = test_ds.df.iloc[i]["urdu_text"]
        roman_true = test_ds.df.iloc[i]["roman_text"]

        src = torch.tensor([sp_urdu.encode(urdu, out_type=int)]).to(DEVICE)
        trg = torch.tensor([sp_roman.encode(roman_true, out_type=int)]).to(DEVICE)

        output = model(src, trg, 0)
        pred_ids = output.argmax(2).cpu().numpy()[0]
        roman_pred = " ".join([sp_roman.id_to_piece(id) for id in pred_ids if id not in [0, sp_roman.pad_id()]])

        print(f"\nUrdu: {urdu}")
        print(f"Target Roman Urdu: {roman_true}")
        print(f"Predicted Roman Urdu: {roman_pred}")

if __name__ == "__main__":
    evaluate_model()
