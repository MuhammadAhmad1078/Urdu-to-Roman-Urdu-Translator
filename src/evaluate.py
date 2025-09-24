# evaluate.py
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import Levenshtein as Lev
import sentencepiece as spm

from dataset import TranslationDataset, collate_fn
from model import Encoder, Decoder, Seq2Seq

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def greedy_decode(model, src, sp_roman, max_len=50):
    """Greedy decoding with attention."""
    enc_out, hidden, cell = model.encoder(src)
    inp = torch.tensor([sp_roman.bos_id()], device=DEVICE)
    result = []
    for _ in range(max_len):
        out, hidden, cell = model.decoder(inp, hidden, cell, enc_out)
        top1 = out.argmax(1).item()
        if top1 == sp_roman.eos_id(): break
        result.append(top1)
        inp = torch.tensor([top1], device=DEVICE)
    return result

def evaluate_model():
    TEST_FILE = os.path.join("data", "processed", "tokenized", "test_tok.jsonl")
    test_ds = TranslationDataset(TEST_FILE, max_len=50)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    vocab_dir = os.path.join("data", "processed", "vocab")
    sp_urdu = spm.SentencePieceProcessor(model_file=os.path.join(vocab_dir, "urdu_bpe.model"))
    sp_roman = spm.SentencePieceProcessor(model_file=os.path.join(vocab_dir, "roman_bpe.model"))

    urdu_vocab_size, roman_vocab_size = sp_urdu.get_piece_size(), sp_roman.get_piece_size()

    encoder = Encoder(urdu_vocab_size, 512, 256, n_layers=2, dropout=0.6)
    decoder = Decoder(roman_vocab_size, 512, 256, n_layers=4, dropout=0.6, tie_embeddings=False)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    smoothie = SmoothingFunction().method4

    bleu_scores, cers, accs, losses, rows = [], [], [], [], []

    with torch.no_grad():
        for src, trg in test_loader:
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            output = model(src, trg, teacher_forcing_ratio=0.0)
            V = output.shape[-1]
            loss = criterion(output[:,1:].reshape(-1,V), trg[:,1:].reshape(-1))
            losses.append(loss.item())

            for i in range(src.size(0)):
                gold_ids = trg[i].cpu().numpy().tolist()
                greedy_ids = greedy_decode(model, src[i].unsqueeze(0), sp_roman)

                def trim(seq, eos): return [int(tok) for tok in seq if tok not in (0, eos)]
                gold_trim, greedy_ids = trim(gold_ids, sp_roman.eos_id()), [int(x) for x in greedy_ids]

                gold_txt, greedy_txt = sp_roman.decode(gold_trim), sp_roman.decode(greedy_ids)

                if gold_trim:
                    bleu = sentence_bleu([gold_txt.split()], greedy_txt.split(), smoothing_function=smoothie)
                    cer = Lev.distance(gold_txt, greedy_txt) / max(1, len(gold_txt))
                    token_acc = np.mean([p == g for p,g in zip(greedy_ids, gold_trim)]) if gold_trim else 0
                    bleu_scores.append(bleu); cers.append(cer); accs.append(token_acc)

                rows.append({"Urdu": sp_urdu.decode(src[i].cpu().numpy().tolist()),
                             "Gold": gold_txt, "Greedy": greedy_txt})

    avg_loss, perplexity = np.mean(losses), np.exp(np.mean(losses))
    avg_bleu, avg_cer, avg_acc = np.mean(bleu_scores), np.mean(cers), np.mean(accs)

    print("\n📊 Test Results:")
    print(f"- Loss: {avg_loss:.4f}")
    print(f"- Perplexity: {perplexity:.4f}")
    print(f"- BLEU (greedy): {avg_bleu:.4f}")
    print(f"- CER  (greedy): {avg_cer:.4f}")
    print(f"- Token Acc: {avg_acc:.4f}")

    os.makedirs("models", exist_ok=True)
    pd.DataFrame(rows).to_csv("models/predictions.csv", index=False)
    print("✅ Predictions saved to models/predictions.csv")

    print("\n🔹 Sample Predictions:")
    for row in rows[:5]:
        print(f"Urdu:   {row['Urdu']}")
        print(f"Gold:   {row['Gold']}")
        print(f"Greedy: {row['Greedy']}\n")

if __name__ == "__main__":
    evaluate_model()
