import torch
from torch import nn
import torch.nn.functional as F

from nltk.tokenize import word_tokenize

from custom_utils import encode_texts, decode_ids
from model_structure import Seq2Seq

device = "cpu"

import json

uz_encode_vocab = json.load(open("uz_encoding_vocab.json", "r"))

uz_decode_vocab = json.load(open("uz_decoding_vocab.json", "r"))
uz_decode_vocab = {int(k): v for k, v in uz_decode_vocab.items()}

en_encode_vocab = json.load(open("en_encoding_vocab.json", "r"))

en_decode_vocab = json.load(open("en_decoding_vocab.json", "r"))
en_decode_vocab = {int(k): v for k, v in en_decode_vocab.items()}

en2uz_model = Seq2Seq(
    input_vocab_size=len(en_encode_vocab),
    output_vocab_size=len(uz_encode_vocab),
    emb_dim=1000,
    hid_size=1000,
).to(device)

en2uz_model.load_state_dict(
    torch.load("en2uz_lstm_model_weights_50k.pth", map_location=device)
)


uz2en_model = Seq2Seq(
    input_vocab_size=len(uz_encode_vocab),
    output_vocab_size=len(en_encode_vocab),
    emb_dim=1000,
    hid_size=1000,
).to(device)

uz2en_model.load_state_dict(
    torch.load("uz2en_lstm_model_weights_50k.pth", map_location=device)
)

import gradio as gr


def swap_labels_and_texts(is_en2uz, source_text, target_text):
    if is_en2uz:
        return "Uzbek", "English", False, target_text, source_text
    else:
        return "English", "Uzbek", True, target_text, source_text


def translate_text(source_text, is_en2uz):
    if is_en2uz:
        input_tokens, input_mask = encode_texts(
            [source_text], en_encode_vocab, decoder_input=False, device=device
        )
        output_tokens, _ = en2uz_model.translate(input_tokens, input_mask, max_len=50)
        target_text = decode_ids(
            output_tokens, uz_decode_vocab, uz_encode_vocab["<eos>"]
        )[0]

    else:
        input_tokens, input_mask = encode_texts(
            [source_text], uz_encode_vocab, decoder_input=False, device=device
        )
        output_tokens, _ = uz2en_model.translate(input_tokens, input_mask, max_len=50)
        target_text = decode_ids(
            output_tokens, en_decode_vocab, en_encode_vocab["<eos>"]
        )[0]
    return target_text


with gr.Blocks() as demo:
    is_en2uz = gr.State(True)

    with gr.Row(equal_height=True):
        label1 = gr.Label(value="English", label="Source Language")
        swap_button = gr.Button("⇄", size="sm", variant="huggingface")
        label2 = gr.Label(value="Uzbek", label="Target Language")

    with gr.Group():
        with gr.Row(equal_height=True):
            source_textbox = gr.Textbox(label="Source Text", placeholder="Enter text")
            target_textbox = gr.Textbox(
                label="Target Text", placeholder="Translation", interactive=False
            )

        clear_button = gr.ClearButton(
            [source_textbox, target_textbox], size="sm", variant="stop"
        )
    translate_button = gr.Button("Translate", variant="primary", size="lg")

    translate_button.click(
        fn=translate_text,
        inputs=[source_textbox, is_en2uz],
        outputs=target_textbox,
    )

    swap_button.click(
        fn=swap_labels_and_texts,
        inputs=[is_en2uz, source_textbox, target_textbox],
        outputs=[label1, label2, is_en2uz, source_textbox, target_textbox],
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
