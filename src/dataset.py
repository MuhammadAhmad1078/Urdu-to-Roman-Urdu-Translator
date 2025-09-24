import json
import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, jsonl_file, max_len=50):
        self.data = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                ur = r["urdu_ids"][:max_len]
                ro = r["roman_ids"][:max_len]
                self.data.append((ur, ro))
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ur, ro = self.data[idx]
        return ur, ro

def collate_fn(batch):
    import torch
    # lists of lists -> tensors, keep true lengths first
    src = [torch.tensor(b[0], dtype=torch.long) for b in batch]
    trg = [torch.tensor(b[1], dtype=torch.long) for b in batch]
    src_lengths = torch.tensor([len(s) for s in src], dtype=torch.long)
    trg_lengths = torch.tensor([len(t) for t in trg], dtype=torch.long)
    src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    trg = torch.nn.utils.rnn.pad_sequence(trg, batch_first=True, padding_value=0)
    return src, trg, src_lengths, trg_lengths
