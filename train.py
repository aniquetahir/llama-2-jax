from functools import partial
import jax
from jax import Array
from jax.experimental.multihost_utils import process_allgather
import jax.numpy as jnp
import jax.random as rand
from jax_smi import initialise_tracking
import math
import optax
import time
from transformers import LlamaTokenizer
from tqdm import tqdm
from typing import Any, Callable, Optional
from lib.model import LlamaModel, Decoder, Attention

from lib.dataloader import LlamaDataLoader
from lib.gsm_data import GSMDataset, TrainData, gsm_collate_fn_train
from lib.loss import cross_entropy_loss
from lib.model import Llama, llama_model, model_config_llama2_7B,llama_model_lora
from lib.multihost_utils import shard_model_params_to_multihost
from lib.param_utils import load_params, save_params
from lib.proc_init_utils import initialise_tpu
from os.path import join as pjoin
import einops as ops
import flax
import tree
from typing import Dict, List, Tuple, Union
from jax.sharding import Mesh, NamedSharding, PartitionSpec, PositionalSharding
from functools import partial
import chex
from collections import namedtuple



optimize: Optional[Callable]
BASE_WEIGHTS_PATH = '/media/anique/Data/projects/llama-weights'
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

cpu_devices = jax.devices('cpu')
cpu_sharding = NamedSharding(Mesh(cpu_devices, ('D',)), PartitionSpec(None))

gpu_devices = jax.devices('gpu')
gpu_sharding_mp = PositionalSharding(gpu_devices)
gpu_sharding_mp = gpu_sharding_mp.reshape((1, len(gpu_devices)))


@jax.value_and_grad
def train_forward_lora(lora_params, lora_config, params: Llama, data_batch: TrainData, *, key: rand.KeyArray):
    seq, seq_mask, labels, labels_mask = data_batch
    outputs = llama_model_lora(lora_params, lora_config, params.model, seq, seq_mask, key=key, model_config=model_config_llama2_7B)
    logits = outputs @ params.lm_head
    loss = cross_entropy_loss(logits, labels, mask=labels_mask)
    return loss

@partial(jax.jit, static_argnames=('lora_config',))
def train_step_lora(lora_params, lora_config, params: Llama, opt_state: Any, total_loss: Array, data_batch: TrainData, key: rand.KeyArray) -> tuple[Llama, Any, Array, Array, rand.KeyArray]:
    key, subkey = rand.split(key)
    loss, grads = train_forward_lora(lora_params, lora_config, params, data_batch, key=subkey)
    total_loss += loss
    updates, opt_state = optimize(grads, opt_state, lora_params)  # type: ignore
    lora_params = optax.apply_updates(lora_params, updates)
    return lora_params, opt_state, total_loss, loss, key

@jax.value_and_grad
def train_forward(params: Llama, data_batch: TrainData, *, key: rand.KeyArray):
    seq, seq_mask, labels, labels_mask = data_batch
    outputs = llama_model(params.model, seq, seq_mask, key=key, model_config=model_config_llama2_7B)
    logits = outputs @ params.lm_head
    loss = cross_entropy_loss(logits, labels, mask=labels_mask)
    return loss

@jax.jit
def train_step(params: Llama, opt_state: Any, total_loss: Array, data_batch: TrainData, key: rand.KeyArray) -> tuple[Llama, Any, Array, Array, rand.KeyArray]:
    key, subkey = rand.split(key)
    loss, grads = train_forward(params, data_batch, key=subkey)
    total_loss += loss
    updates, opt_state = optimize(grads, opt_state, params)  # type: ignore
    params = optax.apply_updates(params, updates)
    return params, opt_state, total_loss, loss, key

def main() -> None:
    global optimize

    # jax.profiler.start_trace('/tmp/tensorboard')

    lr = 0.0001
    batch_size = 1
    n_accumulation_steps = 8
    max_len = 512
    n_epochs = 7
    seed = 3407

    # initialise_tpu('v4-16', n_devices=8, rank=0)
    is_process_0 = jax.process_index() == 0
    cpu_device = jax.devices('cpu')[0]
    gpu_devices = jax.devices('gpu')[0]

    # if is_process_0:
    #     import wandb
    #     wandb.init(project='llama-finetuning-gsm', config=dict(learning_rate=lr, batch_size=batch_size * n_accumulation_steps, n_epochs=n_epochs, optimiser='adamw'))
    #     initialise_tracking()

    key = rand.PRNGKey(seed)
    tokenizer = LlamaTokenizer.from_pretrained(pjoin(BASE_WEIGHTS_PATH, 'llama2-7B'))
    dataset = GSMDataset(split='train')
    collate_fn = partial(gsm_collate_fn_train, tokenizer, max_len)
    dataloader = LlamaDataLoader(dataset, collate_fn, batch_size, seed)

    LoraConfig = namedtuple('LoraConfig', ['LORA_R', 'LORA_ALPHA', 'LORA_DROPOUT'])
    loraConfig = LoraConfig(LORA_R=LORA_R, LORA_ALPHA=LORA_ALPHA, LORA_DROPOUT=LORA_DROPOUT)

    with jax.default_device(gpu_devices):
        params = load_params('llama2-7B.pickle')
        # params_lora = LlamaLoraModel(loraConfig, params)
        pass
    # params = shard_model_params_to_multihost(params)

    # params_flat = tree.flatten(params)

    print('parameters loaded')

    # extract q_proj and v_proj from params
    q_proj_shape, v_proj_shape = params.model.decoder.attention.q_proj.shape, params.model.decoder.attention.v_proj.shape

    # create lora params from q_proj and v_proj
    DB, H, N_REP, N_HEADS, D_K = q_proj_shape
    assert v_proj_shape == (DB, H, N_HEADS, D_K,)

    # create lora params from q_proj and v_proj
    key, split_qa, split_va = rand.split(key, 3)
    q_lora_A = rand.normal(split_qa, (DB, H, N_REP, N_HEADS, LORA_R), dtype=jnp.bfloat16)
    q_lora_B = jnp.zeros((DB, LORA_R, N_REP, N_HEADS, D_K), dtype=jnp.bfloat16)

    v_lora_A = rand.normal(split_va, (DB, H, N_HEADS, LORA_R), dtype=jnp.bfloat16)
    v_lora_B = jnp.zeros((DB, LORA_R, N_HEADS, D_K), dtype=jnp.bfloat16)

    lora_params = {
        'q_lora_A': q_lora_A,
        'q_lora_B': q_lora_B,
        'v_lora_A': v_lora_A,
        'v_lora_B': v_lora_B,
    }


    # jax.profiler.save_device_memory_profile('memory.prof')
    # @jax.jit
    # def merge_lora_params(lora_params: Dict[str, chex.Array], params: chex.Array):
    #     lora_r, lora_alpha = LORA_R, LORA_ALPHA
    #     flat_params = tree.flatten(params)
    #
    #     assert isinstance(lora_params['q_lora_A'], chex.Array)
    #     assert isinstance(lora_params['q_lora_B'], chex.Array)
    #     assert isinstance(lora_params['v_lora_A'], chex.Array)
    #     assert isinstance(lora_params['v_lora_B'], chex.Array)
    #     assert lora_params['q_lora_A'].shape == (DB, H, N_REP, N_HEADS, LORA_R)
    #     assert lora_params['q_lora_B'].shape == (DB, LORA_R, N_REP, N_HEADS, D_K)
    #     assert lora_params['v_lora_A'].shape == (DB, H, N_HEADS, LORA_R)
    #     assert lora_params['v_lora_B'].shape == (DB, LORA_R, N_HEADS, D_K)
    #
    #     q_AB = (lora_alpha/lora_r) * ops.einsum(
    #         lora_params['q_lora_A'],
    #         lora_params['q_lora_B'],
    #         'db h n_rep n_heads r, db r n_rep n_heads d_k -> db h n_rep n_heads d_k')
    #
    #     v_AB = (lora_alpha/lora_r) * ops.einsum(
    #         lora_params['v_lora_A'],
    #         lora_params['v_lora_B'],
    #         'db h n_heads r, db r n_heads d_k -> db h n_heads d_k')
    #
    #     return q_AB + flat_params[2], v_AB + flat_params[4]
    #
    # for i in range(2):
    #     tmp = merge_lora_params(lora_params, params)
    #
    # @jax.jit
    # def insert_q_v_params(q_params: chex.Array, v_params:chex.Array, flat_params:List):
    #     assert isinstance(flat_params, list)
    #     assert len(flat_params) == 12
    #     return (*flat_params[0:2], q_params, flat_params[3], v_params, *flat_params[5:12])
    #
    # @jax.jit
    # def get_merged_params(lora_params: Dict[str, chex.Array], flat_params):
    #     q_params, v_params = merge_lora_params(lora_params, flat_params)
    #     flat_insertion = insert_q_v_params(q_params, v_params, flat_params)
    #     final_params = tree.unflatten_as(params, flat_insertion)
    #     return jax.tree_util.tree_map(lambda x: x.mean(), final_params)
    #
    # print('running lora merge')
    # test_lora_addition = get_merged_params(lora_params, params_flat)
    #
    # print(type(test_lora_addition))
    #
    # exit()
    # params = shard_model_params_to_multihost(params)
    if is_process_0:
        print('Successfully loaded and sharded model parameters!')

    n_steps = math.ceil(len(dataloader) / n_accumulation_steps)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.,
        peak_value=lr,
        warmup_steps=n_steps,
        decay_steps=n_steps + 1,
        end_value=lr,
    )
    optimizer = optax.adamw(learning_rate=schedule)
    optimizer = optax.MultiSteps(optimizer, n_accumulation_steps)
    optimize = optimizer.update

    # opt_state = optimizer.init(params)
    opt_state = optimizer.init(lora_params)


    for _ in range(n_epochs):
        pbar = tqdm(total=len(dataloader) // n_accumulation_steps)
        step_loss = 0.0
        total_loss = jnp.zeros(())

        # if is_process_0:
        #     def report_to_wandb(start_time, opt_state, loss):
        #         nonlocal step_loss
        #         step_loss += loss.item()
        #         if optimizer.has_updated(opt_state):
        #             wandb.log({'train loss': step_loss / n_accumulation_steps, 'time': time.time() - start_time})
        #             step_loss = 0.0
        #             pbar.update()

        for step, data_batch in enumerate(dataloader):
            start_time = time.time()
            lora_params, opt_state, total_loss, loss, key = train_step_lora(lora_params, loraConfig, params, opt_state, total_loss, data_batch, key)
            print(total_loss)
            # if is_process_0:
            #     jax.debug.callback(report_to_wandb, start_time, opt_state, loss)

        # if is_process_0:
        #     wandb.log({'epoch loss': total_loss.item() / (step + 1)})

    gathered_params = process_allgather(lora_params)
    if is_process_0:
        save_params(gathered_params, f'{wandb.run.name}.pickle')  # type: ignore

if __name__ == '__main__':
    main()
