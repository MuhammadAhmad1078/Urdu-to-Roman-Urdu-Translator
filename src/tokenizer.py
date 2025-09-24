import os, json
import pandas as pd
import sentencepiece as spm

PROCESSED_DIR = os.path.join("data", "processed")
TRAIN_FILE = os.path.join(PROCESSED_DIR, "train.csv")
VALID_FILE = os.path.join(PROCESSED_DIR, "valid.csv")
TEST_FILE  = os.path.join(PROCESSED_DIR, "test.csv")

VOCAB_DIR = os.path.join(PROCESSED_DIR, "vocab")
URDU_MODEL  = os.path.join(VOCAB_DIR, "urdu_bpe.model")
ROMAN_MODEL = os.path.join(VOCAB_DIR, "roman_bpe.model")

TOKENIZED_DIR = os.path.join(PROCESSED_DIR, "tokenized")
TRAIN_TOK = os.path.join(TOKENIZED_DIR, "train_tok.jsonl")
VALID_TOK = os.path.join(TOKENIZED_DIR, "valid_tok.jsonl")
TEST_TOK  = os.path.join(TOKENIZED_DIR, "test_tok.jsonl")

def train_bpe(vocab_size_ur=4000, vocab_size_ro=4000, character_coverage=1.0):
    os.makedirs(VOCAB_DIR, exist_ok=True)
    df = pd.read_csv(TRAIN_FILE)
    ur = df["urdu_text"].astype(str).tolist()
    ro = df["roman_text"].astype(str).tolist()

    ur_tmp = os.path.join(VOCAB_DIR, "urdu_tmp.txt")
    ro_tmp = os.path.join(VOCAB_DIR, "roman_tmp.txt")
    open(ur_tmp, "w", encoding="utf-8").write("\n".join(ur))
    open(ro_tmp, "w", encoding="utf-8").write("\n".join(ro))

    spm.SentencePieceTrainer.Train(
        input=ur_tmp, model_prefix=os.path.join(VOCAB_DIR, "urdu_bpe"),
        vocab_size=vocab_size_ur, character_coverage=character_coverage, model_type="bpe"
    )
    spm.SentencePieceTrainer.Train(
        input=ro_tmp, model_prefix=os.path.join(VOCAB_DIR, "roman_bpe"),
        vocab_size=vocab_size_ro, character_coverage=character_coverage, model_type="bpe"
    )
    os.remove(ur_tmp); os.remove(ro_tmp)
    print(f"✅ Urdu BPE: {URDU_MODEL}")
    print(f"✅ Roman BPE: {ROMAN_MODEL}")

def load_tokenizers():
    sp_u = spm.SentencePieceProcessor(model_file=URDU_MODEL)
    sp_r = spm.SentencePieceProcessor(model_file=ROMAN_MODEL)
    print(f"Urdu  BOS/EOS/PAD/UNK: {sp_u.bos_id()}/{sp_u.eos_id()}/{sp_u.pad_id()}/{sp_u.unk_id()}")
    print(f"Roman BOS/EOS/PAD/UNK: {sp_r.bos_id()}/{sp_r.eos_id()}/{sp_r.pad_id()}/{sp_r.unk_id()}")
    return sp_u, sp_r

def tokenize_and_save(df, sp_u, sp_r, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            u_ids = [sp_u.bos_id()] + sp_u.encode(row["urdu_text"], out_type=int) + [sp_u.eos_id()]
            r_ids = [sp_r.bos_id()] + sp_r.encode(row["roman_text"], out_type=int) + [sp_r.eos_id()]
            f.write(json.dumps({"urdu_ids": u_ids, "roman_ids": r_ids}, ensure_ascii=False) + "\n")
    print(f"✅ Saved: {out_path} ({len(df)})")

def coverage(sp_model, texts, lbl=""):
    unk = sp_model.unk_id()
    total = unk_ct = 0
    for t in texts:
        ids = sp_model.encode(t, out_type=int)
        total += len(ids)
        unk_ct += sum(1 for x in ids if x == unk)
    cov = 100 * (1 - unk_ct / max(1, total))
    print(f"🔍 {lbl} coverage: {cov:.2f}%")

def process_all():
    sp_u, sp_r = load_tokenizers()
    tr = pd.read_csv(TRAIN_FILE); va = pd.read_csv(VALID_FILE); te = pd.read_csv(TEST_FILE)
    tokenize_and_save(tr, sp_u, sp_r, TRAIN_TOK)
    tokenize_and_save(va, sp_u, sp_r, VALID_TOK)
    tokenize_and_save(te, sp_u, sp_r, TEST_TOK)
    coverage(sp_u, tr["urdu_text"].astype(str).tolist(), "Urdu")
    coverage(sp_r, tr["roman_text"].astype(str).tolist(), "Roman")

if __name__ == "__main__":
    # bump roman vocab if you still see <unk> drift:
    train_bpe(vocab_size_ur=4000, vocab_size_ro=6000, character_coverage=1.0)
    process_all()
