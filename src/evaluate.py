import os
import torch
import torch.nn as nn
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
# Evaluation
# --------------------------
def evaluate_model():
    VOCAB_DIR = os.path.join("data", "processed", "vocab")
    sp_urdu = spm.SentencePieceProcessor(model_file=os.path.join(VOCAB_DIR, "urdu_bpe.model"))
    sp_roman = spm.SentencePieceProcessor(model_file=os.path.join(VOCAB_DIR, "roman_bpe.model"))

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

            # Encoder pass
            encoder_outputs, hidden, cell = model.encoder(src)

            # Decoder (no teacher forcing)
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
            losses.append(loss.item())

            # Metrics
            preds = outputs.argmax(2).cpu().numpy()
            trgs = trg.cpu().numpy()

            for p, t in zip(preds, trgs):
                pred_tokens = [sp_roman.id_to_piece(i) for i in p if i not in [0, sp_roman.pad_id()]]
                true_tokens = [sp_roman.id_to_piece(i) for i in t if i not in [0, sp_roman.pad_id()]]

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

    # Show qualitative examples
    print("\n🔹 Sample Translations:")
    for i in range(5):
        urdu = test_ds.df.iloc[i]["urdu_text"]
        roman_true = test_ds.df.iloc[i]["roman_text"]

        tokens = [sp_urdu.bos_id()] + sp_urdu.encode(urdu, out_type=int) + [sp_urdu.eos_id()]
        src = torch.tensor(tokens).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            encoder_outputs, hidden, cell = model.encoder(src)

        outputs = [sp_roman.bos_id()]
        input = torch.tensor([outputs[-1]]).to(DEVICE)

        for _ in range(50):
            with torch.no_grad():
                output, hidden, cell = model.decoder(input, hidden, cell, encoder_outputs)

            top1 = output.argmax(1).item()
            outputs.append(top1)

            if top1 == sp_roman.eos_id():
                break
            input = torch.tensor([top1]).to(DEVICE)

        roman_pred = sp_roman.decode(outputs[1:])  # skip <sos>

        print(f"\nUrdu: {urdu}")
        print(f"Target Roman Urdu: {roman_true}")
        print(f"Predicted Roman Urdu: {roman_pred}")

# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    evaluate_model()
