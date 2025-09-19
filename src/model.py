import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# Attention (Luong-style with scaling)
# --------------------------
class Attention(nn.Module):
    def __init__(self, hid_dim, scale=True):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 3, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)
        self.scale = scale
        self.hid_dim = hid_dim

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch, hid_dim]
        # encoder_outputs: [batch, src_len, hid_dim*2]
        src_len = encoder_outputs.shape[1]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch, src_len, hid_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch, src_len, hid_dim]
        attention = self.v(energy).squeeze(2)  # [batch, src_len]

        if self.scale:
            attention = attention / (self.hid_dim ** 0.5)

        return F.softmax(attention, dim=1)  # [batch, src_len]

# --------------------------
# Encoder (BiLSTM + packed sequences)
# --------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            emb_dim, hid_dim, num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_cell = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, lengths=None):
        embedded = self.dropout(self.embedding(src))

        # Pack padded sequence for efficiency
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
            packed_outputs, (hidden, cell) = self.rnn(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        else:
            outputs, (hidden, cell) = self.rnn(embedded)

        # concat last forward + backward states
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)  # [batch, hid_dim*2]
        cell_cat = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)

        hidden_proj = torch.tanh(self.fc(hidden_cat)).unsqueeze(0).repeat(4, 1, 1)
        cell_proj = torch.tanh(self.fc_cell(cell_cat)).unsqueeze(0).repeat(4, 1, 1)

        return outputs, hidden_proj, cell_proj

# --------------------------
# Decoder (LSTM + Attention)
# --------------------------
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=4, dropout=0.3, tie_embeddings=False):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            emb_dim + hid_dim * 2, hid_dim, num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        self.fc_out = nn.Linear(hid_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hid_dim)

        if tie_embeddings:
            if emb_dim != hid_dim + hid_dim * 2:
                print("⚠️ Warning: embedding dim and output projection mismatch, cannot tie.")
            else:
                self.fc_out.weight = self.embedding.weight

        # Special token placeholders (to be set externally in training/eval)
        self.sos_idx = 1
        self.eos_idx = 2

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)  # [batch, 1]
        embedded = self.dropout(self.embedding(input))  # [batch, 1, emb_dim]

        attn_weights = self.attention(hidden[-1], encoder_outputs)  # [batch, src_len]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [batch, 1, hid_dim*2]

        rnn_input = torch.cat((embedded, context), dim=2)  # [batch, 1, emb_dim+hid_dim*2]

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        output = torch.cat((output.squeeze(1), context.squeeze(1)), dim=1)  # [batch, hid_dim+hid_dim*2]
        prediction = self.fc_out(output)  # [batch, output_dim]

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
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src, lengths)
        input = trg[:, 0]  # <sos>

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs
