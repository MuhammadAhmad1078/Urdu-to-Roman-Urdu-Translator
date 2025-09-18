import torch
import torch.nn as nn

# --------------------------
# Encoder (BiLSTM)
# --------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(
            emb_dim, hid_dim, num_layers=n_layers,
            dropout=dropout, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(hid_dim * 2, hid_dim)  # reduce bidirectional output
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)

        # hidden, cell from BiLSTM: [num_layers*2, batch, hid_dim]
        # We only need the last forward and backward hidden states
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden_proj = torch.tanh(self.fc(hidden_cat))  # [batch, hid_dim]

        # Expand to match decoder's 4 layers
        hidden_proj = hidden_proj.unsqueeze(0).repeat(4, 1, 1)
        cell_proj = torch.zeros_like(hidden_proj)

        return outputs, hidden_proj, cell_proj


# --------------------------
# Decoder (LSTM)
# --------------------------
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=4, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(
            emb_dim, hid_dim, num_layers=n_layers,
            dropout=dropout, batch_first=True
        )
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell


# --------------------------
# Seq2Seq
# --------------------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.embedding.num_embeddings

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        _, hidden, cell = self.encoder(src)
        input = trg[:, 0]  # <sos>

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs
