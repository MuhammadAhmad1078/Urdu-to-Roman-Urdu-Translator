import os, torch, torch.nn as nn, pandas as pd, numpy as np
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import Levenshtein as Lev
import sentencepiece as spm

from dataset import TranslationDataset, collate_fn
from model import Encoder, Decoder, Seq2Seq

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def greedy_decode(model, src, src_len, sp_roman, max_len=50):
    enc_out, hidden, cell = model.encoder(src, src_len)
    input_tok = torch.tensor([sp_roman.bos_id()], device=DEVICE)
    result = []
    for _ in range(max_len):
        out, hidden, cell = model.decoder(input_tok, hidden, cell, enc_out)
        top1 = out.argmax(1).item()
        if top1 == sp_roman.eos_id(): break
        result.append(top1); input_tok = torch.tensor([top1], device=DEVICE)
    return result

def evaluate_model():
    TEST_FILE = os.path.join("data", "processed", "tokenized", "test_tok.jsonl")
    test_ds = TranslationDataset(TEST_FILE)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    vocab_dir = os.path.join("data", "processed", "vocab")
    sp_urdu = spm.SentencePieceProcessor(model_file=os.path.join(vocab_dir, "urdu_bpe.model"))
    sp_roman = spm.SentencePieceProcessor(model_file=os.path.join(vocab_dir, "roman_bpe.model"))
    urdu_vocab_size, roman_vocab_size = sp_urdu.get_piece_size(), sp_roman.get_piece_size()

    encoder = Encoder(urdu_vocab_size, 512, 256, n_layers=3, dropout=0.6)
    decoder = Decoder(roman_vocab_size, 512, 256, n_layers=3, dropout=0.7, tie_embeddings=True)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    smoothie = SmoothingFunction().method4

    bleu_scores, cers, accs, losses, rows = [], [], [], [], []

    with torch.no_grad():
        for src, trg, src_len, _ in test_loader:
            src, trg, src_len = src.to(DEVICE), trg.to(DEVICE), src_len.to(DEVICE)
            out = model(src, trg, teacher_forcing_ratio=0.0, lengths=src_len)
            V = out.shape[-1]
            loss = criterion(out[:,1:].reshape(-1,V), trg[:,1:].reshape(-1))
            losses.append(loss.item())

            for i in range(src.size(0)):
                gold_ids = trg[i].cpu().numpy().tolist()
                pred_ids = greedy_decode(model, src[i].unsqueeze(0), src_len[i].unsqueeze(0), sp_roman)

                gold_trim = [int(tok) for tok in gold_ids if tok not in (0, sp_roman.eos_id())]
                pred_ids = [int(x) for x in pred_ids]

                gold_txt, pred_txt = sp_roman.decode(gold_trim), sp_roman.decode(pred_ids)
                if gold_trim:
                    bleu = sentence_bleu([gold_txt.split()], pred_txt.split(), smoothing_function=smoothie)
                    cer = Lev.distance(gold_txt, pred_txt) / max(1, len(gold_txt))
                    token_acc = np.mean([p == g for p,g in zip(pred_ids, gold_trim)]) if gold_trim else 0
                    bleu_scores.append(bleu); cers.append(cer); accs.append(token_acc)

                rows.append({"Urdu": sp_urdu.decode(src[i].cpu().numpy().tolist()),
                             "Gold": gold_txt, "Greedy": pred_txt})

    avg_loss, perplexity = np.mean(losses), np.exp(np.mean(losses))
    print("\n📊 Test Results:")
    print(f"- Loss: {avg_loss:.4f}")
    print(f"- Perplexity: {perplexity:.4f}")
    print(f"- BLEU: {np.mean(bleu_scores):.4f}")
    print(f"- CER: {np.mean(cers):.4f}")
    print(f"- Token Acc: {np.mean(accs):.4f}")

    os.makedirs("models", exist_ok=True)
    pd.DataFrame(rows).to_csv("models/predictions.csv", index=False)
    print("✅ Predictions saved to models/predictions.csv")

    for row in rows[:5]:
        print(f"\nUrdu:   {row['Urdu']}")
        print(f"Gold:   {row['Gold']}")
        print(f"Greedy: {row['Greedy']}")

if __name__ == "__main__":
    evaluate_model()
