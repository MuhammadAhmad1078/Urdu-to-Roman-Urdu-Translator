import os
import json
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import Levenshtein as Lev
import sentencepiece as spm

from dataset import TranslationDataset, collate_fn
from model import Encoder, Decoder, Seq2Seq

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- Beam search with length-norm + trigram blocking + light coverage penalty ---------
def beam_search(model, src, src_len, sp_roman, beam_size=5, max_len=50, alpha=0.7, lambda_cov=0.1):
    model.eval()
    with torch.no_grad():
        enc_out, h, c = model.encoder(src, lengths=src_len)  # enc_out: [1,S,2*hid]
        # each beam item: (seq[tokens], h, c, logprob, cov_vec[S], trigrams_set)
        S = enc_out.size(1)
        sos = model.decoder.sos_idx
        eos = model.decoder.eos_idx

        beams = [(
            [sos], h, c, 0.0, torch.zeros(S, device=DEVICE), set()
        )]
        finished = []

        for _ in range(max_len):
            new_beams = []
            for seq, h_t, c_t, lp, cov, trigs in beams:
                last = torch.tensor([seq[-1]], device=DEVICE)
                if last.item() == eos:
                    finished.append((seq, lp))
                    continue
                # one step
                logits, h_n, c_n = model.decoder(last, h_t, c_t, enc_out)
                logp = torch.log_softmax(logits, dim=-1).squeeze(0)  # [V]

                # trigram blocking
                if len(seq) >= 3:
                    a, b = seq[-2], seq[-1]
                    # ban tokens that would form an already seen trigram (a,b,x)
                    for x in range(logp.size(0)):
                        if (a, b, x) in trigs:
                            logp[x] = -1e9

                # top-k
                topk = torch.topk(logp, beam_size)
                for idx, lp_add in zip(topk.indices.tolist(), topk.values.tolist()):
                    new_seq = seq + [idx]
                    # update coverage: add current attention
                    att_w = model.decoder.attention(h_n[-1], enc_out).squeeze(0)  # [S]
                    cov_new = cov + att_w
                    cov_pen = torch.clamp(att_w, max=1.0).sum().item()  # soft coverage
                    # update trigrams
                    trigs_new = set(trigs)
                    if len(seq) >= 2:
                        trigs_new.add((seq[-2], seq[-1], idx))
                    new_beams.append((new_seq, h_n, c_n, lp + lp_add - lambda_cov * cov_pen, cov_new, trigs_new))

            # prune
            beams = sorted(new_beams, key=lambda x: x[3], reverse=True)[:beam_size]
            if not beams:
                break

        if not finished:
            finished = [(seq, lp) for (seq, _, __, lp, ___, ____) in beams]

        # length-normalized score
        scored = [ (seq, lp / (len(seq) ** alpha)) for (seq, lp) in finished ]
        best = max(scored, key=lambda x: x[1])[0]

        # strip BOS, cut at EOS
        out = []
        for t in best[1:]:
            if t == eos: break
            out.append(t)
        return out

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
    decoder.sos_idx = sp_roman.bos_id()
    decoder.eos_idx = sp_roman.eos_id()
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    smoothie = SmoothingFunction().method4

    losses, bleus, cers, accs = [], [], [], []
    rows = []

    with torch.no_grad():
        for src, trg, src_len, _ in test_loader:
            src, trg, src_len = src.to(DEVICE), trg.to(DEVICE), src_len.to(DEVICE)
            out = model(src, trg, teacher_forcing_ratio=0.0, lengths=src_len)
            V = out.shape[-1]
            loss = criterion(out[:, 1:].reshape(-1, V), trg[:, 1:].reshape(-1))
            losses.append(loss.item())

            # decode each item with strong beam search
            for i in range(src.size(0)):
                best_ids = beam_search(model,
                                       src[i:i+1], src_len[i:i+1],
                                       sp_roman, beam_size=5, max_len=50,
                                       alpha=0.7, lambda_cov=0.1)
                # references
                # trim target (drop BOS, until EOS)
                tgt_ids = []
                for t in trg[i].tolist()[1:]:
                    if t == sp_roman.eos_id() or t == 0: break
                    tgt_ids.append(t)

                pred_text = sp_roman.decode(best_ids)
                true_text = sp_roman.decode(tgt_ids)

                # metrics
                bleu = sentence_bleu([true_text.split()], pred_text.split(), smoothing_function=smoothie)
                cer = Lev.distance(true_text, pred_text) / max(1, len(true_text))
                tok_acc = (sum(p == t for p, t in zip(best_ids, tgt_ids)) / max(1, len(tgt_ids))) if tgt_ids else 0.0

                bleus.append(bleu); cers.append(cer); accs.append(tok_acc)
                rows.append({"true": true_text, "pred": pred_text})

    avg_loss = float(np.mean(losses))
    ppl = math.exp(avg_loss)
    avg_bleu = float(np.mean(bleus)) if bleus else 0.0
    avg_cer  = float(np.mean(cers)) if cers else 1.0
    avg_acc  = float(np.mean(accs)) if accs else 0.0

    print("\n📊 Test Results")
    print(f"- Loss: {avg_loss:.4f}")
    print(f"- Perplexity: {ppl:.4f}")
    print(f"- BLEU: {avg_bleu:.4f}")
    print(f"- CER: {avg_cer:.4f}")
    print(f"- Token Acc: {avg_acc:.4f}")

    os.makedirs("models", exist_ok=True)
    out_path = "models/predictions.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Predictions saved: {out_path}")

    # show a few
    print("\n🔹 Samples")
    for r in rows[:5]:
        print("True:", r["true"])
        print("Pred:", r["pred"])
        print()

if __name__ == "__main__":
    evaluate_model()
