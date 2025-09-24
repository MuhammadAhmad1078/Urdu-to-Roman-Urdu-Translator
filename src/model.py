import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Attention (Luong w/ scaling) ----------
class Attention(nn.Module):
    def __init__(self, hid_dim, scale=True):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 3, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)
        self.scale = scale
        self.hid_dim = hid_dim

    def forward(self, hidden, encoder_outputs):
        # hidden: [B, hid], encoder_outputs: [B, S, 2*hid]
        B, S, _ = encoder_outputs.shape
        hidden = hidden.unsqueeze(1).repeat(1, S, 1)         # [B,S,hid]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [B,S,hid]
        scores = self.v(energy).squeeze(2)                   # [B,S]
        if self.scale:
            scores = scores / (self.hid_dim ** 0.5)
        return F.softmax(scores, dim=1)                      # [B,S]

# ---------- Encoder (BiLSTM + packed) ----------
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=2, dropout=0.6):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            emb_dim, hid_dim, num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True, batch_first=True
        )
        self.fc_h = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_c = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.n_dec_layers = 4  # to match decoder layers

    def forward(self, src, lengths=None):
        emb = self.dropout(self.embedding(src))
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, (h, c) = self.rnn(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            outputs, (h, c) = self.rnn(emb)  # outputs: [B,S,2*hid]

        # last fwd + last bwd
        h_cat = torch.cat((h[-2], h[-1]), dim=1)  # [B,2*hid]
        c_cat = torch.cat((c[-2], c[-1]), dim=1)
        h0 = torch.tanh(self.fc_h(h_cat)).unsqueeze(0).repeat(self.n_dec_layers, 1, 1)  # [L,B,hid]
        c0 = torch.tanh(self.fc_c(c_cat)).unsqueeze(0).repeat(self.n_dec_layers, 1, 1)
        return outputs, h0, c0

# ---------- Decoder (LSTM + Attention, projection for tying) ----------
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=4, dropout=0.6, tie_embeddings=True):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            emb_dim + hid_dim * 2, hid_dim, num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True
        )
        # project (hid + 2*hid) -> emb_dim, then to vocab
        self.pre_out = nn.Linear(hid_dim + hid_dim * 2, emb_dim)
        self.fc_out  = nn.Linear(emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hid_dim)

        if tie_embeddings:
            # now dims match due to projection layer
            self.fc_out.weight = self.embedding.weight

        # SOS/EOS indices for decoders/decoding utils
        self.sos_idx = 1
        self.eos_idx = 2

    def forward(self, input, hidden, cell, encoder_outputs):
        # input: [B], hidden/cell: [L,B,hid], encoder_outputs: [B,S,2*hid]
        inp = input.unsqueeze(1)                              # [B,1]
        emb = self.dropout(self.embedding(inp))               # [B,1,emb]

        attn = self.attention(hidden[-1], encoder_outputs)    # [B,S]
        context = torch.bmm(attn.unsqueeze(1), encoder_outputs)  # [B,1,2*hid]

        rnn_in = torch.cat((emb, context), dim=2)             # [B,1,emb+2*hid]
        out, (hidden, cell) = self.rnn(rnn_in, (hidden, cell))# out: [B,1,hid]

        fused = torch.cat((out.squeeze(1), context.squeeze(1)), dim=1)  # [B,hid+2*hid]
        logits = self.fc_out(torch.tanh(self.pre_out(fused)))           # [B,V]
        return logits, hidden, cell

# ---------- Seq2Seq ----------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5, lengths=None):
        B, T = trg.shape
        V = self.decoder.output_dim
        outputs = torch.zeros(B, T, V, device=self.device)

        enc_out, h, c = self.encoder(src, lengths=lengths)
        inp = trg[:, 0]  # BOS
        for t in range(1, T):
            logits, h, c = self.decoder(inp, h, c, enc_out)
            outputs[:, t] = logits
            teacher = torch.rand(1, device=self.device).item() < teacher_forcing_ratio
            top1 = logits.argmax(1)
            inp = trg[:, t] if teacher else top1
        return outputs
