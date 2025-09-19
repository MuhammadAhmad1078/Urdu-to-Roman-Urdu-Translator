import os
import pandas as pd
import sentencepiece as spm
import json

# --------------------------
# CONFIG (relative paths)
# --------------------------
PROCESSED_DIR = os.path.join("data", "processed")
TRAIN_FILE = os.path.join(PROCESSED_DIR, "train.csv")
VALID_FILE = os.path.join(PROCESSED_DIR, "valid.csv")
TEST_FILE = os.path.join(PROCESSED_DIR, "test.csv")

VOCAB_DIR = os.path.join(PROCESSED_DIR, "vocab")
URDU_MODEL = os.path.join(VOCAB_DIR, "urdu_bpe.model")
ROMAN_MODEL = os.path.join(VOCAB_DIR, "roman_bpe.model")

TOKENIZED_DIR = os.path.join(PROCESSED_DIR, "tokenized")
TRAIN_TOK = os.path.join(TOKENIZED_DIR, "train_tok.jsonl")
VALID_TOK = os.path.join(TOKENIZED_DIR, "valid_tok.jsonl")
TEST_TOK = os.path.join(TOKENIZED_DIR, "test_tok.jsonl")

# --------------------------
# TRAIN SENTENCEPIECE MODELS
# --------------------------
def train_bpe(vocab_size=4000, character_coverage=1.0):
    """
    Train two separate BPE tokenizers:
    - Urdu (source)
    - Roman Urdu (target)
    """
    os.makedirs(VOCAB_DIR, exist_ok=True)

    df = pd.read_csv(TRAIN_FILE)
    urdu_texts = df["urdu_text"].astype(str).tolist()
    roman_texts = df["roman_text"].astype(str).tolist()

    urdu_tmp = os.path.join(VOCAB_DIR, "urdu_tmp.txt")
    roman_tmp = os.path.join(VOCAB_DIR, "roman_tmp.txt")

    with open(urdu_tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(urdu_texts))
    with open(roman_tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(roman_texts))

    # Train Urdu model
    spm.SentencePieceTrainer.Train(
        input=urdu_tmp,
        model_prefix=os.path.join(VOCAB_DIR, "urdu_bpe"),
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type="bpe"
    )

    # Train Roman Urdu model
    spm.SentencePieceTrainer.Train(
        input=roman_tmp,
        model_prefix=os.path.join(VOCAB_DIR, "roman_bpe"),
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type="bpe"
    )

    os.remove(urdu_tmp)
    os.remove(roman_tmp)

    print(f"✅ Urdu BPE model saved: {URDU_MODEL}")
    print(f"✅ Roman Urdu BPE model saved: {ROMAN_MODEL}")

# --------------------------
# LOAD TOKENIZERS
# --------------------------
def load_tokenizers():
    sp_urdu = spm.SentencePieceProcessor(model_file=URDU_MODEL)
    sp_roman = spm.SentencePieceProcessor(model_file=ROMAN_MODEL)
    print("⚙️ Special Tokens:")
    print(f"  Urdu → BOS: {sp_urdu.bos_id()}, EOS: {sp_urdu.eos_id()}, PAD: {sp_urdu.pad_id()}, UNK: {sp_urdu.unk_id()}")
    print(f"  Roman → BOS: {sp_roman.bos_id()}, EOS: {sp_roman.eos_id()}, PAD: {sp_roman.pad_id()}, UNK: {sp_roman.unk_id()}")
    return sp_urdu, sp_roman

# --------------------------
# TOKENIZE DATASETS & SAVE JSON
# --------------------------
def tokenize_and_save(df, urdu_tok, roman_tok, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            urdu_core = urdu_tok.encode(row["urdu_text"], out_type=int)
            roman_core = roman_tok.encode(row["roman_text"], out_type=int)

            urdu_ids = [urdu_tok.bos_id()] + urdu_core + [urdu_tok.eos_id()]
            roman_ids = [roman_tok.bos_id()] + roman_core + [roman_tok.eos_id()]

            record = {"urdu_ids": urdu_ids, "roman_ids": roman_ids}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Saved tokenized dataset: {out_path} ({len(df)} pairs)")

# --------------------------
# COVERAGE CHECK
# --------------------------
def check_vocab_coverage(sp_model, texts, label=""):
    unk_id = sp_model.unk_id()
    total_tokens, unk_tokens = 0, 0
    for line in texts:
        ids = sp_model.encode(line, out_type=int)
        total_tokens += len(ids)
        unk_tokens += sum(1 for t in ids if t == unk_id)
    coverage = 100 * (1 - (unk_tokens / max(1, total_tokens)))
    print(f"🔍 {label} vocab coverage: {coverage:.2f}%")

# --------------------------
# PROCESS ALL SPLITS
# --------------------------
def process_all_splits():
    sp_urdu, sp_roman = load_tokenizers()

    train_df = pd.read_csv(TRAIN_FILE)
    valid_df = pd.read_csv(VALID_FILE)
    test_df = pd.read_csv(TEST_FILE)

    # Tokenize and save
    tokenize_and_save(train_df, sp_urdu, sp_roman, TRAIN_TOK)
    tokenize_and_save(valid_df, sp_urdu, sp_roman, VALID_TOK)
    tokenize_and_save(test_df, sp_urdu, sp_roman, TEST_TOK)

    # Coverage check
    check_vocab_coverage(sp_urdu, train_df["urdu_text"].astype(str).tolist(), "Urdu")
    check_vocab_coverage(sp_roman, train_df["roman_text"].astype(str).tolist(), "Roman Urdu")

# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    # Step 1: Train BPE models
    train_bpe()

    # Step 2: Tokenize datasets and check coverage
    process_all_splits()
