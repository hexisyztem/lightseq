import random
import math
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn


from tests.util import (
    TestDecorator,
    get_fairseq_enc_params,
    get_fairseq_dec_params,
    max_batch_tokens,
    max_seq_len,
    split_custom_layer_grad,
    copy_grad_from_paras,
)

from tests import fairseq_layers
from lightseq.training.ops.pytorch.transformer_encoder_layer_new import (
    LSTransformerEncoderLayer,
)

kt = TestDecorator()


def generate_enc_layer(initial_weights=None, initial_biases=None):
    config = LSTransformerEncoderLayer.get_config(
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        hidden_size=1024,
        intermediate_size=4096,
        nhead=16,
        attn_prob_dropout_ratio=0.0,
        activation_dropout_ratio=0.0,
        hidden_dropout_ratio=0.0,
        pre_layer_norm=True,
        fp16=True,
        local_rank=0,
        activation_fn="relu",
    )
    layer = LSTransformerEncoderLayer(config, initial_weights, initial_biases)
    layer.to(torch.device("cuda:0"), dtype=torch.half)
    return layer


def gen_enc_layer_pair():
    fairseq_enc_layer = fairseq_layers.generate_enc_layer()
    fairseq_enc_layer.train()
    initial_enc_weights, initial_enc_biases = get_fairseq_enc_params(fairseq_enc_layer)
    custom_enc_layer = generate_enc_layer(initial_enc_weights, initial_enc_biases)
    # custom_enc_layer.train()
    return fairseq_enc_layer, custom_enc_layer


@kt.case(dtypes=[torch.half], rtol=1e-3, atol=1e-2, ntest=10)
def test_encoder_layer_forward():
    batch_size, seq_len = kt.bs_sl()
    print(f"(batch_size, seq_len): ({batch_size}, {seq_len})")

    fairseq_enc, custom_enc = gen_enc_layer_pair()

    hidden_states = kt.rand((batch_size, seq_len, 1024))
    self_attn_padding_mask = kt.attn_mask(batch_size, seq_len, dtype=torch.bool)



    def custom():
        inp = hidden_states.clone()
        res = torch.empty_like(inp)
        custom_enc(res, inp, self_attn_padding_mask)
        return [
            res.contiguous().detach(),
        ]

    def baseline():
        res = hidden_states.contiguous().clone()
        res = fairseq_enc(res, self_attn_padding_mask)
        return [
            res.contiguous().detach(),
        ]

    return custom, baseline

if __name__ == "__main__":

    kt.init(device="cuda:0", nhead=16)
    kt.run(
        [
            "test_encoder_layer_forward",
        ]
    )

