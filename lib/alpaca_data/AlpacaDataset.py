import json
import os
from torch.utils.data import Dataset
from typing import Literal, Union
from tqdm import tqdm



def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:\n"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:\n"""

def load_alpaca_data(*, split=Union[Literal['train'], Literal['test']], path=None, split_percent=0.8, tokenizer=None, max_len=512):
    path = os.path.join(path)
    res = []
    with open(path) as f:
        data = json.load(f)

    train_end_idx = int(len(data) * split_percent)
    if split == 'train':
        data = data[:train_end_idx]
    elif split == 'test':
        data = data[train_end_idx:]
    else:
        raise ValueError(f'Invalid split: {split}')

    print('Processing data...')
    for datum in tqdm(data):
        question = generate_prompt(datum['instruction'], datum['input'])
        answer = datum['output']
        # check length of question + answer
        inp_ids = tokenizer(question + answer, add_special_tokens=False, return_attention_mask=False).input_ids
        if len(inp_ids) + 3 > max_len:
            continue

        res.append((question, answer))

    return res


def load_data(*, split=Union[Literal['train'], Literal['test']]):
    path = os.path.join(f'../grade-school-math/grade_school_math/data/{split}.jsonl')
    res = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            question = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n' + data['question'] + '\n\n### Response:\n'
            answer = data['answer']
            answer = answer.replace('#### ', 'Final answer:\n')
            res.append((question, answer))
    return res

class AlpacaDataset(Dataset):
    def __init__(self, *, path: str, split=Union[Literal['train'], Literal['test']], split_percentage=0.8, tokenizer=None, max_len=512) -> None:
        self.data = load_alpaca_data(split=split, path=path, split_percent=split_percentage, tokenizer=tokenizer, max_len=max_len)
        super().__init__()

    def __getitem__(self, idx: int):
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)
