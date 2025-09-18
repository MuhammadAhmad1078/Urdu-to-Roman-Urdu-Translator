import os
import re
import pandas as pd
from urduhack.normalization import normalize_characters, remove_diacritics
from sklearn.model_selection import train_test_split
import json

# --- Project Paths Configuration ---
# Get the base directory of the project, which is the parent of the 'src' folder.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Use os.path.join to construct the correct path to the raw data
ROOT_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'urdu_ghazals_rekhta')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

def clean_urdu_text(text):
    """Normalizes and cleans Urdu text."""
    text = normalize_characters(text)
    text = remove_diacritics(text)
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_roman_urdu_text(text):
    """Normalizes and cleans Roman Urdu text."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_parallel_corpus(root_dir):
    """
    Traverses the directory, pairs Urdu and Roman Urdu ghazals,
    and returns a list of cleaned (Urdu, Roman Urdu) tuples.
    """
    urdu_files = {}
    roman_urdu_files = {}

    for author_dir in os.listdir(root_dir):
        author_path = os.path.join(root_dir, author_dir)
        if os.path.isdir(author_path):
            urdu_path = os.path.join(author_path, 'ur')
            roman_urdu_path = os.path.join(author_path, 'en')
            
            if os.path.exists(urdu_path):
                for filename in os.listdir(urdu_path):
                    urdu_files[filename] = os.path.join(urdu_path, filename)
            
            if os.path.exists(roman_urdu_path):
                for filename in os.listdir(roman_urdu_path):
                    roman_urdu_files[filename] = os.path.join(roman_urdu_path, filename)

    data_pairs = []
    for filename, urdu_filepath in urdu_files.items():
        if filename in roman_urdu_files:
            roman_urdu_filepath = roman_urdu_files[filename]
            
            try:
                with open(urdu_filepath, 'r', encoding='utf-8') as f:
                    urdu_text = f.read()
                with open(roman_urdu_filepath, 'r', encoding='utf-8') as f:
                    roman_urdu_text = f.read()

                urdu_lines = urdu_text.strip().split('\n')
                roman_urdu_lines = roman_urdu_text.strip().split('\n')
                
                if len(urdu_lines) == len(roman_urdu_lines):
                    for urdu_line, roman_urdu_line in zip(urdu_lines, roman_urdu_lines):
                        cleaned_urdu = clean_urdu_text(urdu_line)
                        cleaned_roman_urdu = clean_roman_urdu_text(roman_urdu_line)
                        
                        if cleaned_urdu and cleaned_roman_urdu:
                            data_pairs.append((cleaned_urdu, cleaned_roman_urdu))
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                continue
                
    return data_pairs

# Define special tokens
SOS_token = '<SOS>'
EOS_token = '<EOS>'
PAD_token = '<PAD>'

def build_vocabulary(corpus, special_tokens):
    """
    Builds a character-level vocabulary and mappings.
    """
    chars = set()
    for sentence in corpus:
        if isinstance(sentence, str):
            chars.update(list(sentence))
    
    sorted_chars = sorted(list(chars))
    vocab = special_tokens + sorted_chars
    
    char_to_idx = {char: i for i, char in enumerate(vocab)}
    idx_to_char = {i: char for i, char in enumerate(vocab)}
    
    return vocab, char_to_idx, idx_to_char

# --- Main Execution ---
if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print(f"Attempting to read data from: {ROOT_DIR}")
    
    # --- 1. Create the parallel corpus & clean data ---
    parallel_data = create_parallel_corpus(ROOT_DIR)
    
    if not parallel_data:
        print("No parallel data found. Please check your ROOT_DIR and file structure.")
    else:
        df = pd.DataFrame(parallel_data, columns=['Urdu', 'Roman_Urdu'])
        
        # --- 2. Data Splitting ---
        # 50% for training, 50% for temp (validation + test)
        train_df, temp_df = train_test_split(df, test_size=0.5, random_state=42)
        # Split temp_df into validation (25%) and test (25%)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        # --- 3. Tokenization & Vocabulary Creation ---
        urdu_sentences = df['Urdu'].tolist()
        roman_urdu_sentences = df['Roman_Urdu'].tolist()

        source_special_tokens = [SOS_token, EOS_token, PAD_token]
        target_special_tokens = [SOS_token, EOS_token, PAD_token]
        
        source_vocab, source_char_to_idx, source_idx_to_char = build_vocabulary(urdu_sentences, source_special_tokens)
        target_vocab, target_char_to_idx, target_idx_to_char = build_vocabulary(roman_urdu_sentences, target_special_tokens)

        # --- 4. Save Processed Data & Vocabularies ---
        # Save the split datasets to CSV
        train_df.to_csv(os.path.join(PROCESSED_DIR, "train_data.csv"), index=False)
        val_df.to_csv(os.path.join(PROCESSED_DIR, "validation_data.csv"), index=False)
        test_df.to_csv(os.path.join(PROCESSED_DIR, "test_data.csv"), index=False)
        
        print(f"\nSuccessfully created and saved datasets:")
        print(f" - Training set: {len(train_df)} pairs")
        print(f" - Validation set: {len(val_df)} pairs")
        print(f" - Test set: {len(test_df)} pairs")

        # Save the vocabularies to JSON files
        with open(os.path.join(PROCESSED_DIR, "source_vocab.json"), "w", encoding='utf-8') as f:
            json.dump(source_char_to_idx, f, ensure_ascii=False, indent=4)
        
        with open(os.path.join(PROCESSED_DIR, "target_vocab.json"), "w", encoding='utf-8') as f:
            json.dump(target_char_to_idx, f, ensure_ascii=False, indent=4)
            
        print(f"\nVocabularies saved to '{PROCESSED_DIR}'")
        print(f" - Urdu vocabulary size: {len(source_vocab)}")
        print(f" - Roman Urdu vocabulary size: {len(target_vocab)}")