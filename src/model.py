import torch
import torch.nn as nn

# --------------------------
# Attention (Luong-style, dot + concat)
# --------------------------
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 3, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch, hid_dim]
        # encoder_outputs: [batch, src_len, hid_dim*2] (BiLSTM)
        src_len = encoder_outputs.shape[1]

        # repeat hidden across src_len
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [B, S, H]

        # concat hidden + encoder outputs
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [B,S,H]
        attention = self.v(energy).squeeze(2)  # [B,S]

        return torch.softmax(attention, dim=1)  # normalized attention scores


# --------------------------
# Encoder (BiLSTM)
# --------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(
            emb_dim,
            hid_dim,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(hid_dim * 2, hid_dim)      # reduce bidirectional → hid_dim
        self.fc_cell = nn.Linear(hid_dim * 2, hid_dim) # reduce cell state
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))  # [B,S,E]
        outputs, (hidden, cell) = self.rnn(embedded)  # outputs: [B,S,H*2]

        # concatenate final forward + backward hidden and cell
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)  # [B,H*2]
        cell_cat = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)        # [B,H*2]

        # project down to hid_dim
        hidden_proj = torch.tanh(self.fc(hidden_cat))  # [B,H]
        cell_proj = torch.tanh(self.fc_cell(cell_cat)) # [B,H]

        # expand for decoder layers (n_layers=4)
        hidden_proj = hidden_proj.unsqueeze(0).repeat(4, 1, 1)  # [L,B,H]
        cell_proj = cell_proj.unsqueeze(0).repeat(4, 1, 1)      # [L,B,H]

        return outputs, hidden_proj, cell_proj


# --------------------------
# Decoder (LSTM + Attention)
# --------------------------
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=4, dropout=0.3):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(
            emb_dim + hid_dim * 2,  # input = embedding + context
            hid_dim,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hid_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hid_dim)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)                     # [B,1]
        embedded = self.dropout(self.embedding(input)) # [B,1,E]

        # compute attention
        attn_weights = self.attention(hidden[-1], encoder_outputs) # [B,S]
        attn_weights = attn_weights.unsqueeze(1)                   # [B,1,S]

        context = torch.bmm(attn_weights, encoder_outputs)  # [B,1,H*2]

        # concat embedding + context
        rnn_input = torch.cat((embedded, context), dim=2)  # [B,1,E+H*2]

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        # predict token
        output = torch.cat((output.squeeze(1), context.squeeze(1)), dim=1)  # [B,H+H*2]
        prediction = self.fc_out(output)  # [B,V]

        return prediction, hidden, cell


# --------------------------
# Seq2Seq Wrapper
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
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)
        input = trg[:, 0]  # <sos>

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs
