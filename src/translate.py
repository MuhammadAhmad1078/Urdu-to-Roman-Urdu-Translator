# translate.py
import os
import torch
import sentencepiece as spm
from model import Encoder, Decoder, Seq2Seq

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Load Tokenizers
# --------------------------
VOCAB_DIR = os.path.join("data", "processed", "vocab")
sp_urdu = spm.SentencePieceProcessor(model_file=os.path.join(VOCAB_DIR, "urdu_bpe.model"))
sp_roman = spm.SentencePieceProcessor(model_file=os.path.join(VOCAB_DIR, "roman_bpe.model"))

# --------------------------
# Build Model (match train/eval)
# --------------------------
INPUT_DIM = sp_urdu.get_piece_size()
OUTPUT_DIM = sp_roman.get_piece_size()
EMB_DIM = 512
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3

encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, n_layers=ENC_LAYERS, dropout=0.6)
# ⚠️ Do NOT tie embeddings in this architecture (projection size != emb dim)
decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, n_layers=DEC_LAYERS, dropout=0.7, tie_embeddings=False)
decoder.sos_idx = sp_roman.bos_id()
decoder.eos_idx = sp_roman.eos_id()

model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
model.eval()

# --------------------------
# Greedy Translation
# --------------------------
def translate_sentence(sentence: str, max_len: int = 50) -> str:
    """
    Urdu string -> Roman Urdu (greedy decoding with attention).
    """
    # Encode Urdu with BOS/EOS
    ids = [sp_urdu.bos_id()] + sp_urdu.encode(sentence, out_type=int)[: max_len - 2] + [sp_urdu.eos_id()]
    src = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)              # [1, S]
    lengths = torch.tensor([len(ids)], dtype=torch.long).to(DEVICE)                # [1]

    # Encode
    with torch.no_grad():
        enc_out, hidden, cell = model.encoder(src, lengths)

    # Decode (greedy)
    outputs = []
    inp = torch.tensor([sp_roman.bos_id()], dtype=torch.long, device=DEVICE)       # [1]
    for _ in range(max_len):
        with torch.no_grad():
            logits, hidden, cell = model.decoder(inp, hidden, cell, enc_out)       # logits: [1, V]
        next_id = int(torch.argmax(logits, dim=1).item())
        if next_id == sp_roman.eos_id():
            break
        outputs.append(next_id)
        inp = torch.tensor([next_id], dtype=torch.long, device=DEVICE)

    # Decode to text
    if not outputs:
        return ""
    return sp_roman.decode([int(x) for x in outputs if x != 0])

# --------------------------
# CLI demo
# --------------------------
if __name__ == "__main__":
    urdu_sentence = "محبت کی کوئی زبان نہیں ہوتی"
    print("📝 Urdu:", urdu_sentence)
    print("🔮 Predicted Roman Urdu:", translate_sentence(urdu_sentence))
