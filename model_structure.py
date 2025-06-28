import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = 'cpu'

class Attention_Layer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, enc_out, dec_out, mask):
        """
        enc_out: (batch_size, seq_len, enc_dim)
        dec_out: (batch_size, dec_dim)
        """
        a_t = torch.matmul(enc_out, dec_out.unsqueeze(-1)).squeeze(
            -1
        )  # (batch_size, seq_len)

        mask = mask.clone()
        length = mask.argmin(dim=-1) - 1  # Get the index of the first 0 in each row
        mask[torch.arange(mask.shape[0]), length] = 0
        a_t = a_t.masked_fill(mask == 1, torch.tensor(-1e10))  # (batch_size, seq_len)

        attn_probs = F.softmax(a_t, dim=-1)  # (batch_size, seq_len)

        c_t = torch.matmul(attn_probs.unsqueeze(1), enc_out).squeeze(
            1
        )  # (batch_size, enc_dim)

        return c_t, attn_probs


class Seq2Seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, emb_dim, hid_size):
        super().__init__()
        self.hid_size = hid_size
        self.input_embedding = nn.Embedding(input_vocab_size, emb_dim)
        self.output_embedding = nn.Embedding(output_vocab_size, emb_dim)
        self.dropout = nn.Dropout(0.2)
        self.encoder = nn.LSTM(emb_dim, hid_size, batch_first=True)
        self.decoder = nn.LSTMCell(emb_dim + hid_size, hid_size)
        self.attn = Attention_Layer()
        self.tanh = nn.Tanh()
        self.attentional_layer = nn.Linear(2 * hid_size, hid_size)
        self.output_layer = nn.Linear(hid_size, output_vocab_size)

        self.all_encoder_states = None
        self.input_mask = None

    def encode(self, x):
        x = self.input_embedding(x)  # (batch_size, seq_len, emb_dim)
        x = self.dropout(x)
        all_encoder_states, (last_hidden_state, last_cell_state) = self.encoder(
            x
        )  # (batch_size, seq_len, hid_size)

        last_hidden_state = last_hidden_state.squeeze(0)  # (batch_size, hid_size)
        last_cell_state = last_cell_state.squeeze(0)  # (batch_size, hid_size)

        all_encoder_states = self.dropout(
            all_encoder_states
        )  # Apply dropout to all encoder states
        last_hidden_state = self.dropout(
            last_hidden_state
        )  # Apply dropout to the last hidden

        self.all_encoder_states = all_encoder_states  # (batch_size, seq_len, hid_size)

        first_attentional_state = torch.zeros(
            (x.shape[0], self.hid_size), device=device
        )  # (batch_size, hid_size)

        return (last_hidden_state, last_cell_state, first_attentional_state)

    def decode_step(self, x, prev_output):
        prev_hidden_state, prev_cell_state, prev_attentional_state = prev_output

        x = self.output_embedding(x)  # (batch_size, emb_dim)
        x = torch.cat(
            [x, prev_attentional_state], dim=-1
        )  # (batch_size, emb_dim + hid_size)
        x = self.dropout(x)
        # print(f"x shape: {x.shape}, prev_hidden_state shape: {prev_hidden_state.shape}, prev_cell_state shape: {prev_cell_state.shape}")
        new_hidden_state, new_cell_state = self.decoder(
            x, (prev_hidden_state, prev_cell_state)
        )  # (batch_size, hid_size)
        new_hidden_state = self.dropout(
            new_hidden_state
        )  # Apply dropout to the new hidden state

        c_t, attn_probs = self.attn(
            self.all_encoder_states, new_hidden_state, self.input_mask
        )  # (batch_size,hid_size), (batch_size, seq)

        new_attentional_hid_state = self.tanh(
            self.attentional_layer(torch.cat([c_t, new_hidden_state], dim=-1))
        )  # (batch_size, hid_size)

        output_logits = self.output_layer(
            new_attentional_hid_state
        )  # (batch_size, output_vocab_size)

        return (
            (new_hidden_state, new_cell_state, new_attentional_hid_state),
            output_logits,
            attn_probs,
        )

    def decode(self, x, encoder_output):
        decoder_logits = []
        state = encoder_output

        for i in range(x.shape[1] - 1):
            state, output_logits, _ = self.decode_step(x[:, i], state)
            decoder_logits.append(output_logits)

        return torch.stack(
            decoder_logits, dim=1
        )  # (batch_size, seq_len - 1, output_vocab_size)

    def forward(self, input_tokens, output_tokens, mask):
        self.input_mask = mask

        encoder_output = self.encode(
            input_tokens
        )  # (last_hidden_state, all_encoder_states, mask)
        decoder_output = self.decode(
            output_tokens, encoder_output
        )  # (batch_size, seq_len - 1, output_vocab_size)

        return decoder_output

    def translate(self, input_tokens, input_mask, max_len):
        """
        input_tokens: (batch_size, seq_len)
        max_len: maximum length of the output sequence
        """
        self.eval()
        self.input_mask = input_mask

        decoder_input = torch.zeros(
            input_tokens.shape[0], device=device, dtype=torch.int32
        )  # (batch_size,)
        decoder_output = []
        attn_weights = []

        with torch.no_grad():

            encoder_output = self.encode(input_tokens)  # (batch_size, hid_size)
            state = encoder_output  # (batch_size, hid_size)

            for _ in range(max_len):
                state, output_logits, attn_probs = self.decode_step(
                    decoder_input, state
                )  # (batch_size, hid_size), (batch_size, output_vocab_size) (batch_size, seq_len)
                predicted = output_logits.argmax(dim=-1)  # (batch_size, )
                decoder_output.append(predicted)
                decoder_input = predicted
                attn_weights.append(attn_probs)

        return torch.stack(decoder_output, dim=-1), torch.stack(
            attn_weights, dim=1
        )  # (batch_size, max_len), (batch_size, max_len, seq_len)
