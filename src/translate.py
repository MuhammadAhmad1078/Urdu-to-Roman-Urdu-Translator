import os
import re
import json
import torch
import torch.nn as nn
from urduhack.normalization import normalize_characters, remove_diacritics

# --- Re-define Model Classes ---
# These are necessary to load the saved model's weights correctly.
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        hidden = torch.cat((hidden[::2,:,:], hidden[1::2,:,:]), dim=2)
        cell = torch.cat((cell[::2,:,:], cell[1::2,:,:]), dim=2)
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        enc_outputs, hidden, cell = self.encoder(src)
        input = trg[0,:]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = False # Teacher forcing should be off for inference
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs

# --- Helper Functions for Translation ---

def clean_urdu_text(text):
    """Normalizes and cleans Urdu text."""
    text = normalize_characters(text)
    text = remove_diacritics(text)
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def translate_sentence(sentence, model, source_to_idx, target_to_idx, target_idx_to_char, device, sos_token, eos_token, max_len=50):
    model.eval()
    with torch.no_grad():
        # 1. Preprocess the input sentence
        cleaned_sentence = clean_urdu_text(sentence)
        src_indices = [source_to_idx[char] for char in cleaned_sentence]
        
        # Add SOS and EOS tokens
        src_indices = [source_to_idx[sos_token]] + src_indices + [source_to_idx[eos_token]]
        
        # 2. Convert to a tensor and move to device
        src_tensor = torch.LongTensor(src_indices).unsqueeze(1).to(device)
        
        # 3. Get the encoder's hidden and cell states
        enc_outputs, hidden, cell = model.encoder(src_tensor)
        
        # 4. Prepare for decoding
        # Corrected line: Get the integer index from the char-to-idx map
        input_token = torch.LongTensor([target_to_idx[sos_token]]).to(device)
        
        trg_indices = []
        
        for _ in range(max_len):
            # Pass the last predicted token and hidden states to the decoder
            output, hidden, cell = model.decoder(input_token, hidden, cell)
            
            # Get the index of the predicted character
            pred_token_idx = output.argmax(1).item()
            
            # Stop decoding if we predict the EOS token
            if pred_token_idx == target_to_idx[eos_token]:
                break
            
            # Append the new token index to our output list
            trg_indices.append(pred_token_idx)
            
            # The next input token is the one we just predicted
            input_token = torch.LongTensor([pred_token_idx]).to(device)
            
        # 5. Convert indices back to a Roman Urdu string
        roman_urdu_chars = [target_idx_to_char[i] for i in trg_indices]
        return "".join(roman_urdu_chars)

# --- Main Execution Block ---

if __name__ == "__main__":
    # Define file paths and special tokens
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
    
    SOS_token = '<SOS>'
    EOS_token = '<EOS>'
    PAD_token = '<PAD>'
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load vocabularies
    with open(os.path.join(PROCESSED_DIR, 'source_vocab.json'), 'r', encoding='utf-8') as f:
        source_char_to_idx = json.load(f)
    
    with open(os.path.join(PROCESSED_DIR, 'target_vocab.json'), 'r', encoding='utf-8') as f:
        target_char_to_idx = json.load(f)
    
    # Reverse the target vocabulary to convert indices back to characters
    target_idx_to_char = {v: k for k, v in target_char_to_idx.items()}

    # Define model hyperparameters from training script
    input_dim = len(source_char_to_idx)
    output_dim = len(target_char_to_idx)
    EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    DROPOUT = 0.5

    # Initialize the model and load the trained weights
    enc = Encoder(input_dim, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    dec = Decoder(output_dim, EMB_DIM, HID_DIM * 2, N_LAYERS, DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)

    model_path = os.path.join(MODELS_DIR, 'best_model.pt')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    else:
        print(f"Error: Model file not found at {model_path}")
        exit()

    # --- Interactive Translation ---
    print("\n--- Roman Urdu Transliteration ---")
    print("Enter an Urdu sentence to translate (or 'exit' to quit):")
    
    while True:
        # Read Urdu input from file for reliable input
        import os
        file_path = os.path.join(os.path.dirname(__file__), "test_urdu_input.txt")
        with open(file_path, "r", encoding="utf-8") as f:
            sentence = f.read().strip()
        print(f"Urdu (from file): {sentence}")
        # Optionally, break after one translation
        # Remove or comment out the next two lines if you want to keep looping
        # if sentence.lower() == 'exit':
        #     break

        translation = translate_sentence(
            sentence, 
            model, 
            source_char_to_idx, 
            target_char_to_idx,
            target_idx_to_char, 
            device, 
            SOS_token, 
            EOS_token
        )
        print(f"Roman Urdu: {translation}\n")