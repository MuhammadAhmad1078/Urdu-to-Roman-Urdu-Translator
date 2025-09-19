import argparse
import os
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from tqdm import tqdm
import pandas as pd
import ast
import sentencepiece as spm
from torch.utils.data import DataLoader, Dataset

from model import Encoder, Decoder, Seq2Seq

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Dataset Loader
# --------------------------
class TranslationDataset(Dataset):
    def __init__(self, csv_file, max_len=50):
        self.df = pd.read_csv(csv_file)
        # Filter out malformed rows
        def is_valid(cell):
            if not isinstance(cell, str):
                return False
            s = cell.strip()
            if any(mark in s for mark in [">>>>>>>", "<<<<<<<", "======="]):
                return False
            return s.startswith("[") and s.endswith("]")

        self.df = self.df[self.df["urdu_ids"].apply(is_valid) & self.df["roman_ids"].apply(is_valid)].reset_index(drop=True)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        urdu_ids = ast.literal_eval(self.df.iloc[idx]["urdu_ids"])
        roman_ids = ast.literal_eval(self.df.iloc[idx]["roman_ids"])
        urdu_ids = urdu_ids[:self.max_len]
        roman_ids = roman_ids[:self.max_len]
        return torch.tensor(urdu_ids), torch.tensor(roman_ids)

def collate_fn(batch):
    src, trg = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    trg = nn.utils.rnn.pad_sequence(trg, batch_first=True, padding_value=0)
    return src, trg

# --------------------------
# Training Functions
# --------------------------
def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def evaluate(model, loader, crit, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt, teacher_forcing_ratio=0.0)
            gold = tgt[:, 1:].contiguous()
            loss = crit(logits.view(-1, logits.size(-1)), gold.view(-1))
            total_loss += loss.item()
    return total_loss / max(1, len(loader))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', type=str, default='base', help='experiment name')
    ap.add_argument('--emb', type=int, default=512, help='embedding dimension')
    ap.add_argument('--hid', type=int, default=512, help='hidden dimension')
    ap.add_argument('--enc_layers', type=int, default=2, help='encoder layers')
    ap.add_argument('--dec_layers', type=int, default=4, help='decoder layers')
    ap.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    ap.add_argument('--batch', type=int, default=64, help='batch size')
    ap.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    ap.add_argument('--epochs', type=int, default=60, help='max epochs')
    ap.add_argument('--clip', type=float, default=1.0, help='gradient clipping')
    ap.add_argument('--patience', type=int, default=12, help='early stopping patience')
    ap.add_argument('--scheduler', type=str, default='onecycle', choices=['plateau', 'onecycle'], help='LR scheduler')
    ap.add_argument('--tf_start', type=float, default=0.6, help='initial teacher forcing ratio')
    ap.add_argument('--tf_end', type=float, default=0.25, help='final teacher forcing ratio')
    ap.add_argument('--tf_decay_epochs', type=int, default=20, help='epochs to decay teacher forcing')
    ap.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing')
    ap.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    ap.add_argument('--min_delta', type=float, default=0.01, help='min improvement for early stopping')
    args = ap.parse_args()

    set_seed(42)
    
    # Data paths
    PROCESSED_DIR = os.path.join("data", "processed", "tokenized")
    TRAIN_FILE = os.path.join(PROCESSED_DIR, "train_tok.csv")
    VALID_FILE = os.path.join(PROCESSED_DIR, "valid_tok.csv")
    
    # Load datasets
    train_ds = TranslationDataset(TRAIN_FILE)
    valid_ds = TranslationDataset(VALID_FILE)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    # Load tokenizers for vocab sizes
    vocab_dir = os.path.join("data", "processed", "vocab")
    sp_urdu = spm.SentencePieceProcessor(model_file=os.path.join(vocab_dir, "urdu_bpe.model"))
    sp_roman = spm.SentencePieceProcessor(model_file=os.path.join(vocab_dir, "roman_bpe.model"))
    urdu_vocab_size = sp_urdu.get_piece_size()
    roman_vocab_size = sp_roman.get_piece_size()

    # Model
    encoder = Encoder(urdu_vocab_size, args.emb, args.hid, n_layers=args.enc_layers, dropout=args.dropout)
    decoder = Decoder(roman_vocab_size, args.emb, args.hid, n_layers=args.dec_layers, dropout=args.dropout)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=args.label_smoothing)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    if args.scheduler == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=max(1, len(train_loader)),
            pct_start=0.1,
            div_factor=10.0,
            final_div_factor=100.0
        )
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Experiment tracking
    os.makedirs('runs', exist_ok=True)
    run_dir = os.path.join('runs', args.exp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    best_val = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    print(f"🚀 Starting experiment: {args.exp}")
    print(f"📊 Model: emb={args.emb}, hid={args.hid}, enc_layers={args.enc_layers}, dec_layers={args.dec_layers}")
    print(f"⚙️ Training: lr={args.lr}, batch={args.batch}, dropout={args.dropout}")
    print(f"📈 Scheduler: {args.scheduler}, patience={args.patience}")

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        total_loss = 0.0
        
        # Teacher forcing schedule
        if epoch <= args.tf_decay_epochs:
            tf_ratio = args.tf_start
        else:
            progress = (epoch - args.tf_decay_epochs) / (args.epochs - args.tf_decay_epochs)
            tf_ratio = args.tf_start - (args.tf_start - args.tf_end) * progress
        
        for src, tgt in pbar:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            optimizer.zero_grad()
            
            logits = model(src, tgt, teacher_forcing_ratio=tf_ratio)
            gold = tgt[:, 1:].contiguous()
            loss = criterion(logits.view(-1, logits.size(-1)), gold.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            
            if args.scheduler == 'onecycle':
                scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix(train_loss=f"{total_loss/(pbar.n+1):.4f}", tf_ratio=f"{tf_ratio:.2f}")

        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        val_loss = evaluate(model, valid_loader, criterion, DEVICE)
        val_losses.append(val_loss)
        val_ppl = math.exp(min(20.0, val_loss))
        
        if args.scheduler == 'plateau':
            scheduler.step(val_loss)

        print(f"\nEpoch {epoch} | Train Loss: {train_loss:.4f} | Valid Loss: {val_loss:.4f} | Valid PPL: {val_ppl:.3f}")

        # Early stopping
        if (best_val - val_loss) > args.min_delta:
            best_val = val_loss
            patience_counter = 0
            torch.save({
                'model_state': model.state_dict(),
                'config': vars(args),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epoch': epoch
            }, os.path.join(run_dir, 'best.pt'))
            print("✅ Saved best model")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("⏹️ Early stopping triggered.")
                break

    # Save final results
    results = {
        'best_val_loss': best_val,
        'best_epoch': epoch - patience_counter,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'total_epochs': epoch
    }
    
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎯 Experiment {args.exp} completed!")
    print(f"📊 Best validation loss: {best_val:.4f} at epoch {results['best_epoch']}")
    print(f"📁 Results saved to: {run_dir}")

if __name__ == '__main__':
    main()
