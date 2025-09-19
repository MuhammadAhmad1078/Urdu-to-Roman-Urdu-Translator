import os
import json
import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, jsonl_file, max_len=50):
        self.data = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                urdu_ids = record["urdu_ids"][:max_len]
                roman_ids = record["roman_ids"][:max_len]
                self.data.append((urdu_ids, roman_ids))
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        urdu_ids, roman_ids = self.data[idx]
        return torch.tensor(urdu_ids), torch.tensor(roman_ids)

def collate_fn(batch):
    src, trg = zip(*batch)
    src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    trg = torch.nn.utils.rnn.pad_sequence(trg, batch_first=True, padding_value=0)
    return src, trg
