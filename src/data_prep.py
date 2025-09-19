import os
import re
import csv
import unicodedata
import random
import pandas as pd

# --------------------------
# CONFIG (relative paths)
# --------------------------
RAW_PATH = os.path.join("data", "raw", "urdu_ghazals_rekhta")
PROCESSED_DIR = os.path.join("data", "processed")
FULL_OUTPUT = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
TRAIN_OUTPUT = os.path.join(PROCESSED_DIR, "train.csv")
VALID_OUTPUT = os.path.join(PROCESSED_DIR, "valid.csv")
TEST_OUTPUT = os.path.join(PROCESSED_DIR, "test.csv")

# --------------------------
# ROMAN URDU NORMALIZATION
# --------------------------
roman_char_map = {
    "ā": "a", "ī": "i", "ū": "u",
    "ñ": "n", "ḳ": "k", "ḍ": "d", "ṭ": "t",
    "ṣ": "s", "ż": "z", "ġ": "g",
    "’": "'", "‘": "'", "“": '"', "”": '"'
}

def normalize_roman(text: str) -> str:
    for k, v in roman_char_map.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFKD", text)
    text = "".join([c for c in text if not unicodedata.combining(c)])
    text = text.lower()
    text = re.sub(r"[^a-z\s.,!?']", " ", text)
    text = re.sub(r"(?<=\w)\.(?=\w)", "", text)  # fix ja.ega → jaega
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --------------------------
# URDU NORMALIZATION
# --------------------------
urdu_char_map = {
    "ي": "ی", "ك": "ک", "ۀ": "ہ", "ة": "ہ", "ئ": "ی",
}

def normalize_urdu(text: str) -> str:
    for k, v in urdu_char_map.items():
        text = text.replace(k, v)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"[^\u0600-\u06FF\s.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --------------------------
# MAIN CLEANING PIPELINE
# --------------------------
def build_clean_dataset():
    urdu_lines, roman_lines = [], []

    # Pair files by relative path: traverse English transliteration files
    # and find the corresponding Urdu file by swapping the language segment.
    for root, dirs, files in os.walk(os.path.join(RAW_PATH)):
        for fname in files:
            if fname.startswith("."):
                continue
            file_path = os.path.join(root, fname)

            # Only drive pairing from 'en' side to ensure 1-1 mapping
            if os.sep + "en" + os.sep not in file_path:
                continue

            urdu_path = file_path.replace(os.sep + "en" + os.sep, os.sep + "ur" + os.sep)
            if not os.path.exists(urdu_path):
                # Try 'hi' → some repos may use Hindi script dir; skip if not present
                urdu_path_alt = file_path.replace(os.sep + "en" + os.sep, os.sep + "ur" + os.sep)
                if not os.path.exists(urdu_path_alt):
                    # no matching urdu file; skip
                    continue
                urdu_path = urdu_path_alt

            try:
                with open(file_path, "r", encoding="utf-8") as f_en, \
                     open(urdu_path, "r", encoding="utf-8") as f_ur:
                    en_lines = f_en.read().splitlines()
                    ur_lines = f_ur.read().splitlines()

                    # Align per line; normalize; skip empty pairs after normalization
                    max_len = min(len(en_lines), len(ur_lines))
                    for i in range(max_len):
                        ur_clean = normalize_urdu(ur_lines[i].strip())
                        en_clean = normalize_roman(en_lines[i].strip())
                        if ur_clean and en_clean:
                            urdu_lines.append(ur_clean)
                            roman_lines.append(en_clean)
            except Exception as e:
                print(f"Error pairing {file_path} with {urdu_path}: {e}")

    # Ensure processed folder exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Save full dataset
    with open(FULL_OUTPUT, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["urdu_text", "roman_text"])
        for u, r in zip(urdu_lines, roman_lines):
            writer.writerow([u, r])

    print(f"Full dataset saved: {FULL_OUTPUT} with {len(urdu_lines)} pairs")

    # --------------------------
    # Train/Valid/Test Split
    # --------------------------
    df = pd.DataFrame({"urdu_text": urdu_lines, "roman_text": roman_lines})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    n = len(df)
    n_train = int(0.5 * n)
    n_valid = int(0.25 * n)
    n_test = n - n_train - n_valid

    train_df = df.iloc[:n_train]
    valid_df = df.iloc[n_train:n_train+n_valid]
    test_df = df.iloc[n_train+n_valid:]

    train_df.to_csv(TRAIN_OUTPUT, index=False, encoding="utf-8")
    valid_df.to_csv(VALID_OUTPUT, index=False, encoding="utf-8")
    test_df.to_csv(TEST_OUTPUT, index=False, encoding="utf-8")

    print(f"Train set: {len(train_df)} → {TRAIN_OUTPUT}")
    print(f"Valid set: {len(valid_df)} → {VALID_OUTPUT}")
    print(f"Test set: {len(test_df)} → {TEST_OUTPUT}")


if __name__ == "__main__":
    build_clean_dataset()
