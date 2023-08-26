import fire
import jax
import jax.numpy as jnp
from lib.param_utils import load_params
import pickle
import tree

cpu_device = jax.devices('cpu')[0]
gpu_device = jax.devices('gpu')[0]

def merge_q_v(path, value, q_additive, v_additive):
    if 'q_proj' in path:
        return value + q_additive
    elif 'v_proj' in path:
        return value + v_additive
    else:
        return value

def merge_lora(params_path:str, lora_path:str, output_path:str):
    # load the complete model parameters
    with jax.default_device(cpu_device):
        params = load_params(params_path)

    # load the lora parameters
    with open(lora_path, 'rb') as f:
        lora_params = pickle.load(f)

    print('loaded params (full + lora)')
    print('merging params...')

    # multiply lora components to get final params
    q_lora = jnp.einsum('dmosr,drosn->dmosn', lora_params['q_lora_A'], lora_params['q_lora_B'])
    v_lora = jnp.einsum('dmsr,drsn->dmsn', lora_params['v_lora_A'], lora_params['v_lora_B'])

    # add the final additive params to the original params
    updated_params = tree.map_structure_with_path(lambda p, v: merge_q_v(p, v, q_lora, v_lora), params)

    print('finished merging params')

    with open(output_path, 'wb') as f:
        pickle.dump(updated_params, f)


if __name__ == "__main__":
    fire.Fire(merge_lora)

