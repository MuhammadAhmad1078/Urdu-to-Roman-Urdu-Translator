import os
import time
import math
import random
import json
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Define the Encoder, Decoder, and Seq2Seq classes here for a single, self-contained script.
# For a larger project, you would import these from a separate model.py file.
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
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs

# --- Custom Dataset and DataLoader ---

class TranslationDataset(Dataset):
    def __init__(self, dataframe, source_to_idx, target_to_idx):
        self.df = dataframe
        self.source_to_idx = source_to_idx
        self.target_to_idx = target_to_idx
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        urdu_text = self.df.iloc[idx]['Urdu']
        roman_urdu_text = self.df.iloc[idx]['Roman_Urdu']
        
        # Tokenize and convert to indices
        urdu_indices = [self.source_to_idx[char] for char in urdu_text]
        roman_urdu_indices = [self.target_to_idx[char] for char in roman_urdu_text]
        
        return torch.tensor(urdu_indices, dtype=torch.long), torch.tensor(roman_urdu_indices, dtype=torch.long)

def collate_fn(batch, pad_token_id):
    src_list = [item[0] for item in batch]
    trg_list = [item[1] for item in batch]
    
    # Pad sequences to the same length
    src_padded = nn.utils.rnn.pad_sequence(src_list, batch_first=False, padding_value=pad_token_id)
    trg_padded = nn.utils.rnn.pad_sequence(trg_list, batch_first=False, padding_value=pad_token_id)
    
    return src_padded, trg_padded

# --- Helper Functions for Training ---

def train(model, dataloader, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    
    for i, (src, trg) in enumerate(dataloader):
        src = src.to(device)
        trg = trg.to(device)
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, (src, trg) in enumerate(dataloader):
            src = src.to(device)
            trg = trg.to(device)
            
            output = model(src, trg, 0) # Turn off teacher forcing
            
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
    return epoch_loss / len(dataloader)

# --- Main Execution Block ---

if __name__ == "__main__":
    # Define file paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

    # Load data and vocabularies
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'train_data.csv'))
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'validation_data.csv'))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'test_data.csv'))
    
    # Corrected section: specify encoding for JSON files
    with open(os.path.join(PROCESSED_DIR, 'source_vocab.json'), 'r', encoding='utf-8') as f:
        source_char_to_idx = json.load(f)
    
    with open(os.path.join(PROCESSED_DIR, 'target_vocab.json'), 'r', encoding='utf-8') as f:
        target_char_to_idx = json.load(f)
    
    # Define vocabulary dimensions
    input_dim = len(source_char_to_idx)
    output_dim = len(target_char_to_idx)

    # Define model hyperparameters
    EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    DROPOUT = 0.5
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    N_EPOCHS = 10
    CLIP = 1
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create dataset and dataloader
    pad_token_id = target_char_to_idx['<PAD>']
    
    train_dataset = TranslationDataset(train_df, source_char_to_idx, target_char_to_idx)
    val_dataset = TranslationDataset(val_df, source_char_to_idx, target_char_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_token_id))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=lambda b: collate_fn(b, pad_token_id))
    
    # Corrected Model Initialization
    # Pass the correct number of layers to the Decoder constructor
    enc = Encoder(input_dim, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    dec = Decoder(output_dim, EMB_DIM, HID_DIM * 2, N_LAYERS, DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)

    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
    # Training loop
    best_valid_loss = float('inf')
    
    # Create the 'models' directory if it doesn't exist
    model_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        train_loss = train(model, train_loader, optimizer, criterion, CLIP, device)
        valid_loss = evaluate(model, val_loader, criterion, device)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pt'))