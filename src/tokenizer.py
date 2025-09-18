import os
import pandas as pd
import sentencepiece as spm

# --------------------------
# CONFIG
# --------------------------
PROCESSED_DIR = os.path.join("data", "processed")
TRAIN_FILE = os.path.join(PROCESSED_DIR, "train.csv")
VALID_FILE = os.path.join(PROCESSED_DIR, "valid.csv")
TEST_FILE = os.path.join(PROCESSED_DIR, "test.csv")

VOCAB_DIR = os.path.join(PROCESSED_DIR, "vocab")
URDU_MODEL = os.path.join(VOCAB_DIR, "urdu_bpe.model")
ROMAN_MODEL = os.path.join(VOCAB_DIR, "roman_bpe.model")

TOKENIZED_DIR = os.path.join(PROCESSED_DIR, "tokenized")
TRAIN_TOK = os.path.join(TOKENIZED_DIR, "train_tok.csv")
VALID_TOK = os.path.join(TOKENIZED_DIR, "valid_tok.csv")
TEST_TOK = os.path.join(TOKENIZED_DIR, "test_tok.csv")

# --------------------------
# TRAIN SENTENCEPIECE MODELS
# --------------------------
def train_bpe(vocab_size=8000, character_coverage=1.0):
    """
    Train two separate BPE tokenizers:
    - Urdu (source)
    - Roman Urdu (target)
    """
    os.makedirs(VOCAB_DIR, exist_ok=True)

    df = pd.read_csv(TRAIN_FILE)
    urdu_texts = df["urdu_text"].astype(str).tolist()
    roman_texts = df["roman_text"].astype(str).tolist()

    # Save temp files for SentencePiece training
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

    # Remove temp files
    os.remove(urdu_tmp)
    os.remove(roman_tmp)

    print(f"Urdu BPE model saved: {URDU_MODEL}")
    print(f"Roman Urdu BPE model saved: {ROMAN_MODEL}")

# --------------------------
# LOAD TOKENIZERS
# --------------------------
def load_tokenizers():
    sp_urdu = spm.SentencePieceProcessor(model_file=URDU_MODEL)
    sp_roman = spm.SentencePieceProcessor(model_file=ROMAN_MODEL)
    return sp_urdu, sp_roman

# --------------------------
# TOKENIZE DATASETS
# --------------------------
def tokenize_and_save(df, urdu_tok, roman_tok, out_path):
    tokenized_data = []
    for _, row in df.iterrows():
        urdu_ids = urdu_tok.encode(row["urdu_text"], out_type=int)
        roman_ids = roman_tok.encode(row["roman_text"], out_type=int)
        tokenized_data.append({
            "urdu_ids": urdu_ids,
            "roman_ids": roman_ids
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(tokenized_data).to_csv(out_path, index=False)
    print(f"Saved tokenized dataset: {out_path} ({len(tokenized_data)} pairs)")

def process_all_splits():
    urdu_tok, roman_tok = load_tokenizers()

    train_df = pd.read_csv(TRAIN_FILE)
    valid_df = pd.read_csv(VALID_FILE)
    test_df = pd.read_csv(TEST_FILE)

    tokenize_and_save(train_df, urdu_tok, roman_tok, TRAIN_TOK)
    tokenize_and_save(valid_df, urdu_tok, roman_tok, VALID_TOK)
    tokenize_and_save(test_df, urdu_tok, roman_tok, TEST_TOK)

# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    # Step 1: Train BPE models
    train_bpe()

    # Step 2: Tokenize datasets
    process_all_splits()
