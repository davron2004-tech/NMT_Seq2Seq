import torch
import torch.nn as nn
import numpy as np

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize


def encode_texts(texts, vocab, decoder_input, device):
    ids_padded = []
    mask_padded = []
    texts_split = [word_tokenize(text.lower()) for text in texts]

    max_len = max(map(len, texts_split)) + 1  # +1 for EOS token

    for text in texts_split:
        if decoder_input:
            text_ids = [vocab["<bos>"]]  # Start with BOS token
            mask = [0]
        else:
            text_ids = []
            mask = []
        for i in range(max_len):
            if i < len(text):
                text_ids.append(vocab.get(text[i], vocab["<unk>"]))
                mask.append(0)
            else:
                text_ids.append(vocab["<eos>"])
                mask.append(1)
        if not decoder_input:
            text_ids.reverse()
            mask.reverse()
        ids_padded.append(text_ids)
        mask_padded.append(mask)

    return torch.tensor(ids_padded, device=device), torch.tensor(
        mask_padded, device=device
    )


def decode_ids(ids, vocab, eos_id):
    ids = ids.cpu().numpy() if isinstance(ids, torch.Tensor) else ids
    texts = []
    for id_seq in ids:
        text = []
        for id in id_seq:
            if id == eos_id:
                break
            text.append(vocab.get(id, "<unk>"))
        texts.append(" ".join(text))

    return texts


def compute_bleu(reference_texts, candidate_texts):
    bleu_scores = []

    for ref_text, cand_text in zip(reference_texts, candidate_texts):
        reference_tokens = word_tokenize(ref_text.lower())
        candidate_tokens = word_tokenize(cand_text.lower())

        smoother = SmoothingFunction().method1  # You can try method2, method4 as well

        bleu = (
            sentence_bleu(
                [reference_tokens], candidate_tokens, smoothing_function=smoother
            )
            * 100
        )
        bleu_scores.append(bleu)

    return np.mean(bleu_scores)
