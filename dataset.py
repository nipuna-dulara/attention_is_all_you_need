import torch
import torch.nn as nn
import torch.utils.data as Dataset


class BilingualDataset(Dataset.Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, lang_src, lang_tgt, seq_len):
        super().__init__()
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.ds = ds
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.seq_len = seq_len

        self.sos_token = torch.tensor(
            [self.tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor(
            [self.tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor(
            [self.tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_tgt_pair = self.ds[idx]

        src_text = src_tgt_pair['translation'][self.lang_src]
        tgt_text = src_tgt_pair['translation'][self.lang_tgt]
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_pad_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_pad_tokens = self.seq_len - len(dec_input_tokens) - 1

        if (enc_num_pad_tokens < 0) or (dec_num_pad_tokens < 0):
            raise Exception("Sequence length is too small")

        encoder_input_tokens = torch.cat([self.sos_token, torch.tensor(
            enc_input_tokens, dtype=torch.int64), self.eos_token, torch.tensor([self.pad_token] * enc_num_pad_tokens, dtype=torch.int64)])
        decoder_input_tokens = torch.cat([self.sos_token, torch.tensor(
            dec_input_tokens, dtype=torch.int64), torch.tensor([self.pad_token] * dec_num_pad_tokens, dtype=torch.int64)])
        label = torch.cat([torch.tensor(dec_input_tokens, dtype=torch.int64), self.eos_token, torch.tensor(
            [self.pad_token] * dec_num_pad_tokens, dtype=torch.int64)])

        assert encoder_input_tokens.shape == decoder_input_tokens.shape == label.shape == (
            self.seq_len,)
        return {
            "encoder_input": encoder_input_tokens,
            "decoder_input": decoder_input_tokens,
            # (1,1,seq_len)
            "encoder_mask": (encoder_input_tokens != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            # (1,seq_len) & (1,seq_len, seq_len)
            "decoder_mask": (decoder_input_tokens != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input_tokens.shape[0]),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }


def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
