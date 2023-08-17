from functools import partial
import jax
from jax import Array
import jax.random as rand
import math
from typing import Any, NamedTuple, Optional

from ..rand_utils import split_key_nullable
from .attention import Attention, attention, attention_lora, check_attention, init_attention
from .ModelConfig import ModelConfig
from .dropout import dropout
from .rms_norm import check_rms_norm, init_rms_norm, rms_norm

class DecoderBlock(NamedTuple):
    input_norm: Any  # Array
    attention: Attention
    post_attn_norm: Any  # Array
    gate_proj: Any  # Array
    up_proj: Any  # Array
    down_proj: Any  # Array

# class LoraDecoderBlock(NamedTuple):
#     input_norm: Any  # Array
#     attention: LoraAttention
#     post_attn_norm: Any  # Array
#     gate_proj: Any  # Array
#     up_proj: Any  # Array
#     down_proj: Any  # Array

def check_decoder_block(params: DecoderBlock, *, model_config: ModelConfig) -> None:
    assert isinstance(params.input_norm, Array)
    assert isinstance(params.attention, Attention)
    assert isinstance(params.post_attn_norm, Array)
    assert isinstance(params.gate_proj, Array)
    assert isinstance(params.up_proj, Array)
    assert isinstance(params.down_proj, Array)

    check_rms_norm(params.input_norm, model_config=model_config)
    check_attention(params.attention, model_config=model_config)
    check_rms_norm(params.post_attn_norm, model_config=model_config)
    assert params.gate_proj.shape == (model_config.d_model, model_config.d_ff)
    assert params.up_proj.shape == (model_config.d_model, model_config.d_ff)
    assert params.down_proj.shape == (model_config.d_ff, model_config.d_model)

def init_decoder_block(*, key: rand.KeyArray, model_config: ModelConfig) -> DecoderBlock:
    upper = 1. / math.sqrt(model_config.d_model)
    key0, key1, key2, key3 = rand.split(key, num=4)
    input_norm = init_rms_norm(model_config=model_config)
    attention = init_attention(key=key0, model_config=model_config)
    post_attn_norm = init_rms_norm(model_config=model_config)
    gate_proj = rand.truncated_normal(key1, -upper, upper, (model_config.d_model, model_config.d_ff))
    up_proj = rand.truncated_normal(key2, -upper, upper, (model_config.d_model, model_config.d_ff))
    down_proj = rand.truncated_normal(key3, -upper, upper, (model_config.d_ff, model_config.d_model))
    return DecoderBlock(input_norm, attention, post_attn_norm, gate_proj, up_proj, down_proj)

@partial(jax.jit, static_argnames=('model_config',))
def decoder_block(params: DecoderBlock, seq: Array, attn_mask: Array, *, key: Optional[rand.KeyArray], model_config: ModelConfig) -> Array:
    key0, key1, key2 = split_key_nullable(key, num=3)

    seq_ = seq
    seq = rms_norm(params.input_norm, seq, model_config=model_config)
    seq = attention(params.attention, seq, seq, attn_mask, model_config=model_config)
    seq = dropout(seq, key=key0, model_config=model_config)
    seq += seq_

    seq_ = seq
    seq = rms_norm(params.post_attn_norm, seq, model_config=model_config)
    ff = jax.nn.silu(seq @ params.gate_proj) * (seq @ params.up_proj)
    ff = dropout(ff, key=key1, model_config=model_config)
    seq = ff @ params.down_proj
    seq = dropout(seq, key=key2, model_config=model_config)
    seq += seq_

    return seq

def decoder_block_lora(lora_params, lora_config, params: DecoderBlock, seq: Array, attn_mask: Array, *, key: Optional[rand.KeyArray], model_config: ModelConfig) -> Array:
    key0, key1, key2 = split_key_nullable(key, num=3)

    seq_ = seq
    seq = rms_norm(params.input_norm, seq, model_config=model_config)
    seq = attention_lora(lora_params, lora_config, params.attention, seq, seq, attn_mask, model_config=model_config)
    seq = dropout(seq, key=key0, model_config=model_config)
    seq += seq_

    seq_ = seq
    seq = rms_norm(params.post_attn_norm, seq, model_config=model_config)
    ff = jax.nn.silu(seq @ params.gate_proj) * (seq @ params.up_proj)
    ff = dropout(ff, key=key1, model_config=model_config)
    seq = ff @ params.down_proj
    seq = dropout(seq, key=key2, model_config=model_config)
    seq += seq_

    return seq
