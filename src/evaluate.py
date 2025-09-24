import os
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
# Dataset Loader (JSONL)
# --------------------------
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, max_len=50):
        # Load JSON lines
        self.df = pd.read_json(json_file, lines=True)
        self.max_len = max_len

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        urdu_ids = self.df.iloc[idx]["urdu_ids"]
        roman_ids = self.df.iloc[idx]["roman_ids"]
        return torch.tensor(urdu_ids[:self.max_len]), torch.tensor(roman_ids[:self.max_len])

def collate_fn(batch):
    src, trg = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    trg = nn.utils.rnn.pad_sequence(trg, batch_first=True, padding_value=0)
    return src, trg

# --------------------------
# Decoding Strategies
# --------------------------
def greedy_decode(model, src, sp_roman, max_len=50):
    enc_out, hidden, cell = model.encoder(src)
    input_tok = torch.tensor([sp_roman.bos_id()], device=DEVICE)
    result = []
    for _ in range(max_len):
        out, hidden, cell = model.decoder(input_tok, hidden, cell, enc_out)
        top1 = out.argmax(1).item()
        if top1 == sp_roman.eos_id(): break
        result.append(top1)
        input_tok = torch.tensor([top1], device=DEVICE)
    return result

def beam_search(model, src, sp_roman, beam_size=5, max_len=50):
    enc_out, hidden, cell = model.encoder(src)
    beams = [( [sp_roman.bos_id()], hidden, cell, 0.0 )]
    completed = []
    for _ in range(max_len):
        new_beams = []
        for tokens, h, c, score in beams:
            if tokens[-1] == sp_roman.eos_id():
                completed.append((tokens, score)); continue
            inp = torch.tensor([tokens[-1]], device=DEVICE)
            out, h_new, c_new = model.decoder(inp, h, c, enc_out)
            log_probs = torch.log_softmax(out, dim=-1).squeeze(0)
            topk = torch.topk(log_probs, beam_size)
            for idx, val in zip(topk.indices.tolist(), topk.values.tolist()):
                new_beams.append((tokens+[idx], h_new, c_new, score+val))
        beams = sorted(new_beams, key=lambda x: x[3], reverse=True)[:beam_size]
        if len(completed) >= beam_size: break
    if not completed: completed = beams
    best_seq = max(completed, key=lambda x: x[1])[0]
    return [tok for tok in best_seq if tok not in (sp_roman.bos_id(), sp_roman.eos_id())]

def nucleus_sampling(model, src, sp_roman, top_p=0.9, max_len=50):
    enc_out, hidden, cell = model.encoder(src)
    input_tok = torch.tensor([sp_roman.bos_id()], device=DEVICE)
    result = []
    for _ in range(max_len):
        out, hidden, cell = model.decoder(input_tok, hidden, cell, enc_out)
        probs = torch.softmax(out, dim=-1).squeeze(0).cpu().numpy()
        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = np.sort(probs)[::-1]
        cumsum = np.cumsum(sorted_probs)
        cutoff = np.where(cumsum > top_p)[0][0]
        top_idx = sorted_idx[:cutoff+1]
        top_probs = sorted_probs[:cutoff+1] / sorted_probs[:cutoff+1].sum()
        next_tok = int(np.random.choice(top_idx, p=top_probs))  # ✅ ensure int
        if next_tok == sp_roman.eos_id(): break
        result.append(next_tok)
        input_tok = torch.tensor([next_tok], device=DEVICE)
    return result

# --------------------------
# Evaluation
# --------------------------
def evaluate_model():
    TEST_FILE = os.path.join("data", "processed", "tokenized", "test_tok.jsonl")
    test_ds = TranslationDataset(TEST_FILE)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    vocab_dir = os.path.join("data", "processed", "vocab")
    sp_urdu = spm.SentencePieceProcessor(model_file=os.path.join(vocab_dir, "urdu_bpe.model"))
    sp_roman = spm.SentencePieceProcessor(model_file=os.path.join(vocab_dir, "roman_bpe.model"))
    urdu_vocab_size = sp_urdu.get_piece_size()
    roman_vocab_size = sp_roman.get_piece_size()

    encoder = Encoder(urdu_vocab_size, 512, 256, n_layers=2, dropout=0.5)
    decoder = Decoder(roman_vocab_size, 512, 256, n_layers=4, dropout=0.5)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    smoothie = SmoothingFunction().method4

    bleu_scores, cers, accs, losses, rows = [], [], [], [], []

    with torch.no_grad():
        for src, trg in test_loader:
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            output = model(src, trg, 0)
            out_dim = output.shape[-1]
            loss = criterion(output[:,1:].reshape(-1, out_dim), trg[:,1:].reshape(-1))
            losses.append(loss.item())

            for i in range(src.size(0)):
                gold_ids = trg[i].cpu().numpy().tolist()

                # Use only greedy decoding (best for speed and reliability)
                greedy_ids = greedy_decode(model, src[i].unsqueeze(0), sp_roman)

                def trim(seq, eos): 
                    return [int(tok) for tok in seq if tok not in (0, eos)]

                gold_trim = trim(gold_ids, sp_roman.eos_id())
                greedy_ids = [int(x) for x in greedy_ids]

                gold_txt   = sp_roman.decode(gold_trim)
                greedy_txt = sp_roman.decode(greedy_ids)

                if gold_trim:
                    bleu = sentence_bleu([gold_txt.split()], greedy_txt.split(), smoothing_function=smoothie)
                    cer = Lev.distance(gold_txt, greedy_txt) / max(1, len(gold_txt))
                    token_acc = np.mean([p == g for p,g in zip(greedy_ids, gold_trim)]) if gold_trim else 0
                    bleu_scores.append(bleu); cers.append(cer); accs.append(token_acc)

                rows.append({
                    "Gold": gold_txt,
                    "Greedy": greedy_txt
                })

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

    os.makedirs("models", exist_ok=True)
    pd.DataFrame(rows).to_csv("models/predictions.csv", index=False)
    print("✅ Predictions saved to models/predictions.csv")

    print("\n🔹 Sample Predictions:")
    for row in rows[:5]:
        print(f"Gold:   {row['Gold']}")
        print(f"Greedy: {row['Greedy']}\n")

if __name__ == "__main__":
    evaluate_model()
