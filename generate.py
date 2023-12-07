# from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))
from lib.proc_init_utils import initialise_tpu
# from lib.proc_init_utils import initialise_gpu; initialise_gpu()

import jax
import jax.numpy as jnp
import jax.random as rand
from jax_smi import initialise_tracking
from transformers import LlamaTokenizer

# from lib.generation import TopKGenerationConfig, top_k
from lib.generation import TopPGenerationConfig, top_p
from lib.model import model_config_llama2_7B
from lib.multihost_utils import shard_array_to_multihost, shard_model_params_to_multihost
from lib.param_utils import load_params
from lib.seeding import BEST_INTEGER

from os.path import join as pjoin

BASE_WEIGHTS_PATH = '/media/anique/Data/projects/llama-weights'
MODEL_PATH = 'phase2_params/merged.pkl'
SEQ_LENGTH = 1024

tokenizer = LlamaTokenizer.from_pretrained(pjoin(BASE_WEIGHTS_PATH, 'llama2-7B'))
tokenizer.pad_token = tokenizer.eos_token  # TODO: verify this
query = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You have been provided with a CSV file containing a social media conversation.
For this task, you should only make assumptions about posters based on the provided CSV input. The posts in the input are
in the order of date posted i.e. replies do not occur before posts being replied to.
 Answer the following questions:
a) Summarize the discourse revolving around the active post ID.
b) Is the active post considered bullying?
c) Is the active post considered anti-bullying?
d) If it is bullying, explain why?
e) If it is anti-bullying, explain why?
f) If it is neither, explain why?


### Input:
```post_id,comment,active,reply_to
79299000,>diversity is bad  >diverse areas are the ones where white people are the happiest   rly maeks u think :^),0,
79299160,">>79299000  No it doesn't, it really doesn't, I'll gift you 100 million diversity.",0,
79299551,>>79299000  What's up with those blue places in the middle and north east?,0,
79300343,>>79299551  All the areas in red you have to buy your water bottled or filter it at home. Blue areas have good tap water.  Really makes you think.,0,
79301198,>>79299000  It isn't considered suicide if minorities kill you.,0,
79301291,>>79299000  Diversity areas are the ones where white men don't need to kill themselves because the niggers will do it for them.,1,
```

### Response:"""

sentences = [query]
#
# sentences = [
#     'I believe the meaning of life is',
#     'Simply put, the theory of relativity states that',
#     'Thus, leveraging the potential of quantum computing, we can optimize complex algorithms, paving the way for breakthroughs in fields ranging from cryptography to molecular modeling',
# ]

def main() -> None:
    # initialise_tpu('v4-16', n_devices=8, rank=0)
    is_process_0 = jax.process_index() == 0
    if is_process_0:
        print(jax.devices())
    initialise_tracking()

    key = rand.PRNGKey(BEST_INTEGER)
    cpu_device = jax.devices('cpu')[0]
    with jax.default_device(cpu_device):
        params = load_params(MODEL_PATH)
    # params = shard_model_params_to_multihost(params)

    # top_k_config = TopKGenerationConfig(eos_token_id=tokenizer.eos_token_id, max_length=128, top_k=10)
    top_p_config = TopPGenerationConfig(eos_token_id=tokenizer.eos_token_id, max_length=SEQ_LENGTH, top_p=0.9)

    inputs = tokenizer(sentences, max_length=top_p_config.max_length, padding='max_length', return_tensors='jax')
    seq = inputs.input_ids.astype(jnp.uint16)
    attn_mask = inputs.attention_mask.astype(jnp.bool_)

    seq = shard_array_to_multihost(seq, ...)
    attn_mask = shard_array_to_multihost(attn_mask, ...)

    key, subkey = rand.split(key)
    config_llama2_7B_ = model_config_llama2_7B._replace(dropout_rate=None)
    # generated_seq = top_k(params, seq, attn_mask, key=subkey, model_config=model_config_llama1_7B, top_k_config=top_k_config)
    generated_seq = top_p(params, seq, attn_mask, key=subkey, model_config=config_llama2_7B_, top_p_config=top_p_config)
    decoded_texts = tokenizer.batch_decode(generated_seq, skip_special_tokens=True)

    if is_process_0:
        for decoded_text in decoded_texts:
            print(decoded_text, end='\n\n')

if __name__ == '__main__':
    main()
