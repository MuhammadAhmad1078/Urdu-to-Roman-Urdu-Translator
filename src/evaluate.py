import os
import json
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import Levenshtein as Lev
import sentencepiece as spm

from dataset import TranslationDataset, collate_fn
from model import Encoder, Decoder, Seq2Seq

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Beam Search Decoding
# --------------------------
def beam_search_decode(model, src, max_len=50, beam_size=5):
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src)
        batch_size = src.size(0)

        beams = [[(torch.tensor([model.decoder.sos_idx], device=DEVICE), hidden, 0.0)] for _ in range(batch_size)]
        results = [[] for _ in range(batch_size)]

        for _ in range(max_len):
            new_beams = [[] for _ in range(batch_size)]
            for b in range(batch_size):
                for seq, h, score in beams[b]:
                    if seq[-1].item() == model.decoder.eos_idx:
                        results[b].append((seq, score))
                        continue
                    logits, h_new = model.decoder(seq[-1].unsqueeze(0).unsqueeze(0), h, encoder_outputs[b].unsqueeze(0))
                    log_probs = torch.log_softmax(logits.squeeze(0), dim=-1)
                    topk = torch.topk(log_probs, beam_size)
                    for idx, log_p in zip(topk.indices, topk.values):
                        new_seq = torch.cat([seq, idx.unsqueeze(0)])
                        new_beams[b].append((new_seq, h_new, score + log_p.item()))
                new_beams[b] = sorted(new_beams[b], key=lambda x: x[2], reverse=True)[:beam_size]
            beams = new_beams

        for b in range(batch_size):
            if not results[b]:
                results[b] = beams[b]
            results[b] = sorted(results[b], key=lambda x: x[1], reverse=True)[:1]

        decoded = [seq[0].tolist() for seq in [r[0] for r in results]]
        return decoded

# --------------------------
# Top-k / Nucleus Sampling
# --------------------------
def sample_decode(model, src, max_len=50, top_k=10, top_p=0.9):
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src)
        inputs = torch.tensor([[model.decoder.sos_idx]], device=DEVICE)
        outputs = []

        for _ in range(max_len):
            logits, hidden = model.decoder(inputs[:, -1:], hidden, encoder_outputs)
            logits = logits.squeeze(1)
            probs = torch.softmax(logits, dim=-1)

            # Top-k filtering
            if top_k > 0:
                values, indices = torch.topk(probs, top_k)
                probs = torch.zeros_like(probs).scatter_(1, indices, values)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            # Nucleus filtering
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative_probs > top_p
            sorted_probs[mask] = 0
            probs = torch.zeros_like(probs).scatter_(1, sorted_indices, sorted_probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)

            next_token = torch.multinomial(probs, 1)
            outputs.append(next_token.item())
            inputs = torch.cat([inputs, next_token], dim=1)

            if next_token.item() == model.decoder.eos_idx:
                break

        return outputs

# --------------------------
# Evaluation
# --------------------------
def evaluate_model():
    TEST_FILE = os.path.join("data", "processed", "tokenized", "test_tok.jsonl")
    test_ds = TranslationDataset(TEST_FILE, max_len=50)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    vocab_dir = os.path.join("data", "processed", "vocab")
    sp_urdu = spm.SentencePieceProcessor(model_file=os.path.join(vocab_dir, "urdu_bpe.model"))
    sp_roman = spm.SentencePieceProcessor(model_file=os.path.join(vocab_dir, "roman_bpe.model"))
    urdu_vocab_size = sp_urdu.get_piece_size()
    roman_vocab_size = sp_roman.get_piece_size()

    encoder = Encoder(urdu_vocab_size, 512, 256, n_layers=2, dropout=0.6)
    decoder = Decoder(roman_vocab_size, 512, 256, n_layers=4, dropout=0.6, tie_embeddings=True)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    bleu_scores, cers, accs, losses = [], [], [], []
    smoothie = SmoothingFunction().method4
    predictions = []

    with torch.no_grad():
        for src, trg in test_loader:
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            output = model(src, trg, 0)

            # Loss
            output_dim = output.shape[-1]
            out_flat = output[:,1:].reshape(-1, output_dim)
            trg_flat = trg[:,1:].reshape(-1)
            loss = criterion(out_flat, trg_flat)
            losses.append(loss.item())

            # Greedy predictions for accuracy
            preds = output.argmax(2).cpu().numpy()
            trgs = trg.cpu().numpy()

            for p, t in zip(preds, trgs):
                def trim(seq, eos_id):
                    out = []
                    for tok in seq[1:]:
                        if tok == eos_id:
                            break
                        if tok != 0:
                            out.append(int(tok))
                    return out

                p_trim = trim(p, sp_roman.eos_id())
                t_trim = trim(t, sp_roman.eos_id())

                if not t_trim: 
                    continue

                pred_text = sp_roman.decode(p_trim)
                true_text = sp_roman.decode(t_trim)

                bleu = sentence_bleu([true_text.split()], pred_text.split(), smoothing_function=smoothie)
                bleu_scores.append(bleu)

                cer = Lev.distance(true_text, pred_text) / max(1, len(true_text))
                cers.append(cer)

                token_acc = sum([pp == tt for pp, tt in zip(p_trim, t_trim)]) / max(1, len(t_trim))
                accs.append(token_acc)

                predictions.append({"true": true_text, "pred": pred_text})

    avg_loss = np.mean(losses)
    perplexity = np.exp(avg_loss)
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    avg_cer = np.mean(cers) if cers else 1.0
    avg_acc = np.mean(accs) if accs else 0.0

    print("\n📊 Test Results:")
    print(f"- Loss: {avg_loss:.4f}")
    print(f"- Perplexity: {perplexity:.4f}")
    print(f"- BLEU: {avg_bleu:.4f}")
    print(f"- CER: {avg_cer:.4f}")
    print(f"- Token Acc: {avg_acc:.4f}")

    # Save predictions for error analysis
    out_path = "models/predictions.csv"
    pd.DataFrame(predictions).to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Predictions saved to {out_path}")

    # Show sample predictions
    print("\n🔹 Sample Predictions:")
    for row in predictions[:5]:
        print(f"True: {row['true']}")
        print(f"Pred: {row['pred']}\n")

if __name__ == "__main__":
    evaluate_model()
