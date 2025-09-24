import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
import Levenshtein as Lev
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sentencepiece as spm

from model import Encoder, Decoder, Seq2Seq
from dataset import TranslationDataset, collate_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Greedy Decode (with EOS stop)
# --------------------------
def greedy_decode(model, src, sp_roman, max_len=50):
    enc_out, hidden, cell = model.encoder(src)
    input_tok = torch.tensor([sp_roman.bos_id()], device=DEVICE)
    result = []
    for _ in range(max_len):
        out, hidden, cell = model.decoder(input_tok, hidden, cell, enc_out)
        top1 = out.argmax(1).item()
        if top1 == sp_roman.eos_id():
            break
        result.append(top1)
        input_tok = torch.tensor([top1], device=DEVICE)
    return result

# --------------------------
# Evaluation
# --------------------------
def evaluate_model():
    # Load dataset
    TEST_FILE = os.path.join("data", "processed", "tokenized", "test_tok.jsonl")
    test_ds = TranslationDataset(TEST_FILE, max_len=50)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Tokenizers
    vocab_dir = os.path.join("data", "processed", "vocab")
    sp_urdu = spm.SentencePieceProcessor(model_file=os.path.join(vocab_dir, "urdu_bpe.model"))
    sp_roman = spm.SentencePieceProcessor(model_file=os.path.join(vocab_dir, "roman_bpe.model"))
    urdu_vocab_size = sp_urdu.get_piece_size()
    roman_vocab_size = sp_roman.get_piece_size()

    # Model (match training)
    encoder = Encoder(urdu_vocab_size, 512, 256, n_layers=3, dropout=0.6)
    decoder = Decoder(roman_vocab_size, 512, 256, n_layers=3, dropout=0.7, tie_embeddings=True)
    decoder.sos_idx = sp_roman.bos_id()
    decoder.eos_idx = sp_roman.eos_id()

    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    smoothie = SmoothingFunction().method4

    bleu_scores, cers, accs, losses, rows = [], [], [], [], []

    with torch.no_grad():
        for src, trg in test_loader:
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            output = model(src, trg, 0)  # no teacher forcing
            V = output.shape[-1]
            loss = criterion(output[:, 1:].reshape(-1, V), trg[:, 1:].reshape(-1))
            losses.append(loss.item())

            for i in range(src.size(0)):
                gold_ids = trg[i].cpu().numpy().tolist()
                pred_ids = greedy_decode(model, src[i].unsqueeze(0), sp_roman)

                def trim(seq, eos): 
                    return [int(tok) for tok in seq if tok not in (0, eos)]

                gold_trim = trim(gold_ids, sp_roman.eos_id())
                pred_trim = trim(pred_ids, sp_roman.eos_id())

                gold_txt = sp_roman.decode(gold_trim)
                pred_txt = sp_roman.decode(pred_trim)

                if gold_trim:
                    bleu = sentence_bleu([gold_txt.split()], pred_txt.split(), smoothing_function=smoothie)
                    cer = Lev.distance(gold_txt, pred_txt) / max(1, len(gold_txt))
                    token_acc = np.mean([p == g for p, g in zip(pred_trim, gold_trim)]) if gold_trim else 0
                    bleu_scores.append(bleu); cers.append(cer); accs.append(token_acc)

                rows.append({
                    "Urdu": sp_urdu.decode(trim(src[i].cpu().numpy().tolist(), sp_urdu.eos_id())),
                    "Gold": gold_txt,
                    "Greedy": pred_txt
                })

    # Final metrics
    avg_loss = np.mean(losses)
    perplexity = np.exp(avg_loss)
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    avg_cer = np.mean(cers) if cers else 1.0
    avg_acc = np.mean(accs) if accs else 0.0

    print("\n📊 Test Results:")
    print(f"- Loss: {avg_loss:.4f}")
    print(f"- Perplexity: {perplexity:.4f}")
    print(f"- BLEU (greedy): {avg_bleu:.4f}")
    print(f"- CER  (greedy): {avg_cer:.4f}")
    print(f"- Token Acc: {avg_acc:.4f}")

    # Save predictions
    os.makedirs("models", exist_ok=True)
    pd.DataFrame(rows).to_csv("models/predictions.csv", index=False, encoding="utf-8")
    print("✅ Predictions saved to models/predictions.csv")

    # Show samples
    print("\n🔹 Sample Predictions:")
    for row in rows[:5]:
        print(f"Urdu:   {row['Urdu']}")
        print(f"Gold:   {row['Gold']}")
        print(f"Greedy: {row['Greedy']}\n")

if __name__ == "__main__":
    evaluate_model()
