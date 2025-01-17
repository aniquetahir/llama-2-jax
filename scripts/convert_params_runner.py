from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

import fire
import jax
import jax.numpy as jnp
from transformers import LlamaForCausalLM

from lib.model import check_llama, model_config_llama1_7B, model_config_llama2_70B, model_config_llama2_7B, model_config_llama2_13B
from lib.param_utils import convert_llama, save_params
from os.path import join as pjoin

SAVE_DIR = '/media/anique/Data/projects/llama_jax_weights'
BASE_PATH = '/media/anique/Data/projects/llama-weights'
pairs = {
    'llama1-7B':  (pjoin(BASE_PATH, 'llama1-7B'), model_config_llama1_7B),
    'llama2-7B':  (pjoin(BASE_PATH, 'llama2-7B'), model_config_llama2_7B),
    'llama2-13B': (pjoin(BASE_PATH, 'llama2-13B'), model_config_llama2_13B),
    'llama2-70B': (pjoin(BASE_PATH, 'llama2-70B'), model_config_llama2_70B),
}

def convert(target: str) -> None:
    path, model_config = pairs[target]
    model_pt = LlamaForCausalLM.from_pretrained(path)  # cannot use `torch_dtype=torch.bfloat16` here, see https://github.com/pytorch/pytorch/issues/101781
    params = convert_llama(model_pt, model_config=model_config)
    params = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    check_llama(params, model_config=model_config)
    save_params(params, pjoin(SAVE_DIR, f'{target}.pickle'))

if __name__ == '__main__':
  fire.Fire(convert)
