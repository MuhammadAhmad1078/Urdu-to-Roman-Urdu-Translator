import torch
import torch.nn as nn

# --------------------------
# Attention (Luong-style)
# --------------------------
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 3, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch, hid_dim]
        # encoder_outputs: [batch, src_len, hid_dim*2]
        src_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch, src_len, hid_dim]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attn_weights = self.v(energy).squeeze(2)  # [batch, src_len]

        return torch.softmax(attn_weights, dim=1)


# --------------------------
# Encoder (BiLSTM with packing)
# --------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=3, dropout=0.6):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(
            emb_dim, hid_dim, num_layers=n_layers,
            dropout=dropout, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_cell = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, lengths):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        cell_cat = torch.cat((cell[-2], cell[-1]), dim=1)

        hidden_proj = torch.tanh(self.fc(hidden_cat)).unsqueeze(0).repeat(hidden.size(0), 1, 1)
        cell_proj = torch.tanh(self.fc_cell(cell_cat)).unsqueeze(0).repeat(cell.size(0), 1, 1)

        return outputs, hidden_proj, cell_proj


# --------------------------
# Decoder (LSTM + Attention)
# --------------------------
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=3, dropout=0.7, tie_embeddings=False):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(
            emb_dim + hid_dim * 2, hid_dim, num_layers=n_layers,
            dropout=dropout, batch_first=True
        )
        self.fc_out = nn.Linear(hid_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hid_dim)

        if tie_embeddings:
            if emb_dim == self.fc_out.in_features:
                self.fc_out.weight = self.embedding.weight
            else:
                print("⚠️ Embedding dim mismatch, cannot tie embeddings.")

        self.sos_idx = None
        self.eos_idx = None

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)  # [batch, 1]
        embedded = self.dropout(self.embedding(input))

        attn_weights = self.attention(hidden[-1], encoder_outputs).unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)

        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        output = torch.cat((output.squeeze(1), context.squeeze(1)), dim=1)
        prediction = self.fc_out(output)

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

    def forward(self, src, trg, teacher_forcing_ratio=0.5, lengths=None):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src, lengths)
        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs
