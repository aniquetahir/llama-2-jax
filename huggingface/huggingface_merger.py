import jax
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM
from torch import nn as tnn
import numpy as np
import jax.numpy as jnp
import torch

from lib.param_utils import load_params

LLAMA_HF_7B_PATH = '/media/anique/Data/projects/llama-weights/llama2-7B'
JAX_ORIG_MODEL_PATH = '../llama2-7B.pickle'
JAX_PHASE2_PARAMS_PATH = '../phase2_params/merged.pkl'
SAVE_PATH = '/media/anique/Data/projects/llama-weights/llama2-7B-merged'

def lorize_huggingface_llama():
    # load lora merged params in jax format
    with jax.default_device(jax.devices('cpu')[0]):
        jax_params = load_params(JAX_PHASE2_PARAMS_PATH)

    model = LlamaForCausalLM.from_pretrained(LLAMA_HF_7B_PATH)
    print('model loaded')
    # load the huggingface model

    q_projs = []
    for name, weight in model.named_parameters():
        if 'q_proj' in name:
            q_projs.append((name, weight,))

    v_projs = []
    for name, weight in model.named_parameters():
        if 'v_proj' in name:
            v_projs.append((name, weight,))


    jax_qproj = jax_params.model.decoder.attention.q_proj
    # replace the q_proj weights with the lora merged weights
    for i, (name, param) in enumerate(q_projs):
        # get the shape
        shape = param.shape
        # get corresponding jax param
        jax_q_i = jax_qproj[i].reshape(shape[0], -1).T.astype(jnp.float32)
        jax_q_i = torch.from_numpy(np.asarray(jax_q_i))
        param_q_i = tnn.Parameter(jax_q_i)
        model.model.layers[i].self_attn.q_proj.weight = param_q_i

    jax_vproj = jax_params.model.decoder.attention.v_proj
    # replace the v_proj weights with the lora merged weights
    for i, (name, param) in enumerate(v_projs):
        shape = param.shape
        jax_v_i = jax_vproj[i].reshape(shape[0], -1).T.astype(jnp.float32)
        jax_v_i = torch.from_numpy(np.asarray(jax_v_i))
        param_v_i = tnn.Parameter(jax_v_i)
        model.model.layers[i].self_attn.v_proj.weight = param_v_i



    # save the model seperately
    model.save_pretrained(SAVE_PATH)
    print(f'model saved to {SAVE_PATH}')
    # replace the model parameters with the lora merged params

    # save the updated model parameters
    pass

if __name__ == "__main__":
    with torch.no_grad():
        lorize_huggingface_llama()


