import os, json
import torch
import torch.nn as nn
import pandas as pd

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, max_len=50):
        self.df = pd.read_json(json_file, lines=True)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        urdu_ids = self.df.iloc[idx]["urdu_ids"][:self.max_len]
        roman_ids = self.df.iloc[idx]["roman_ids"][:self.max_len]
        return torch.tensor(urdu_ids), torch.tensor(roman_ids)

def collate_fn(batch):
    src, trg = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    trg = nn.utils.rnn.pad_sequence(trg, batch_first=True, padding_value=0)
    return src, trg
