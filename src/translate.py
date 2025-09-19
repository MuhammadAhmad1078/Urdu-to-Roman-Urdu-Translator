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
# Load Model
# --------------------------
INPUT_DIM = sp_urdu.get_piece_size()
OUTPUT_DIM = sp_roman.get_piece_size()
HID_DIM = 512

encoder = Encoder(INPUT_DIM, 512, HID_DIM, n_layers=2, dropout=0.5)
decoder = Decoder(OUTPUT_DIM, 512, HID_DIM, n_layers=4, dropout=0.5)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
model.eval()

# --------------------------
# Translate Function
# --------------------------
def translate_sentence(sentence, max_len=50, no_repeat_ngram_size=3):
    tokens = [sp_urdu.bos_id()] + sp_urdu.encode(sentence, out_type=int)[:max_len-2] + [sp_urdu.eos_id()]
    src_tensor = torch.tensor(tokens).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)

    outputs = [sp_roman.bos_id()]
    input = torch.tensor([outputs[-1]]).to(DEVICE)

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden, cell = model.decoder(input, hidden, cell, encoder_outputs)

        # Block repeated n-grams
        logits = output.squeeze(0)
        if len(outputs) >= no_repeat_ngram_size:
            n = no_repeat_ngram_size
            prev_ngram = tuple(outputs[-(n-1):])
            # Penalize tokens forming an already seen n-gram
            for i in range(1, len(outputs)-n+2):
                if tuple(outputs[i:i+n-1]) == prev_ngram:
                    ban_token = outputs[i+n-1] if i+n-1 < len(outputs) else None
                    if ban_token is not None:
                        logits[ban_token] = -1e9

        top1 = int(torch.argmax(logits).item())
        outputs.append(top1)

        if top1 == sp_roman.eos_id():
            break
        input = torch.tensor([top1]).to(DEVICE)

    translation = sp_roman.decode([int(i) for i in outputs[1:]])  # skip <sos>
    return translation

# --------------------------
# Example Usage
# --------------------------
if __name__ == "__main__":
    urdu_sentence = "محبت کی کوئی زبان نہیں ہوتی"
    print("📝 Urdu:", urdu_sentence)
    print("🔮 Predicted Roman Urdu:", translate_sentence(urdu_sentence))
