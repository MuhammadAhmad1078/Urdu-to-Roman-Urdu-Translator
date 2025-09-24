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
# Load Model (match training)
# --------------------------
INPUT_DIM = sp_urdu.get_piece_size()
OUTPUT_DIM = sp_roman.get_piece_size()
HID_DIM = 256  # match train/eval
EMB_DIM = 512

encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, n_layers=3, dropout=0.6)
decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, n_layers=3, dropout=0.7, tie_embeddings=True)
decoder.sos_idx = sp_roman.bos_id()
decoder.eos_idx = sp_roman.eos_id()

model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
model.eval()

# --------------------------
# Greedy Translation
# --------------------------
def translate_sentence(sentence, max_len=50):
    # Encode Urdu sentence into IDs with BOS/EOS
    tokens = [sp_urdu.bos_id()] + sp_urdu.encode(sentence, out_type=int)[:max_len-2] + [sp_urdu.eos_id()]
    src_tensor = torch.tensor(tokens).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        enc_out, hidden, cell = model.encoder(src_tensor)

    outputs = []
    input_tok = torch.tensor([sp_roman.bos_id()], device=DEVICE)

    for _ in range(max_len):
        with torch.no_grad():
            out, hidden, cell = model.decoder(input_tok, hidden, cell, enc_out)

        top1 = out.argmax(1).item()
        if top1 == sp_roman.eos_id():
            break
        outputs.append(top1)
        input_tok = torch.tensor([top1], device=DEVICE)

    # Decode prediction IDs back to Roman Urdu text
    decoded = [int(x) for x in outputs if x != 0]
    return sp_roman.decode(decoded)

# --------------------------
# Example Usage
# --------------------------
if __name__ == "__main__":
    urdu_sentence = "محبت کی کوئی زبان نہیں ہوتی"
    print("📝 Urdu:", urdu_sentence)
    print("🔮 Predicted Roman Urdu:", translate_sentence(urdu_sentence))
