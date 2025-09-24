# translate.py
import os
import torch
import sentencepiece as spm
from model import Encoder, Decoder, Seq2Seq

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB_DIR = os.path.join("data", "processed", "vocab")
sp_urdu = spm.SentencePieceProcessor(model_file=os.path.join(VOCAB_DIR, "urdu_bpe.model"))
sp_roman = spm.SentencePieceProcessor(model_file=os.path.join(VOCAB_DIR, "roman_bpe.model"))

INPUT_DIM, OUTPUT_DIM = sp_urdu.get_piece_size(), sp_roman.get_piece_size()
EMB_DIM, HID_DIM = 512, 256

encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, n_layers=2, dropout=0.6)
decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, n_layers=4, dropout=0.6, tie_embeddings=False)
decoder.sos_idx, decoder.eos_idx = sp_roman.bos_id(), sp_roman.eos_id()

model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
model.eval()

def translate_sentence(sentence: str, max_len: int = 50) -> str:
    ids = [sp_urdu.bos_id()] + sp_urdu.encode(sentence, out_type=int)[: max_len-2] + [sp_urdu.eos_id()]
    src = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        enc_out, hidden, cell = model.encoder(src)

    outputs, inp = [], torch.tensor([sp_roman.bos_id()], dtype=torch.long, device=DEVICE)
    for _ in range(max_len):
        with torch.no_grad():
            logits, hidden, cell = model.decoder(inp, hidden, cell, enc_out)
        top1 = int(logits.argmax(1).item())
        if top1 == sp_roman.eos_id(): break
        outputs.append(top1)
        inp = torch.tensor([top1], dtype=torch.long, device=DEVICE)

    return sp_roman.decode([int(x) for x in outputs if x != 0])

if __name__ == "__main__":
    urdu_sentence = "محبت کی کوئی زبان نہیں ہوتی"
    print("📝 Urdu:", urdu_sentence)
    print("🔮 Predicted Roman Urdu:", translate_sentence(urdu_sentence))
