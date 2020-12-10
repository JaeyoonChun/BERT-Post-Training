from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch
import json
import random
from utils import clean
import copy 

class BuildDataset(Dataset):
    def __init__(self, data, tokenizer, domain, mode=None):
        self.data = data
        self.tokenizer = tokenizer
        self.domain = domain
        self.mode = mode

    def __getitem__(self, idx):
        if self.domain == 'source':
            text = self.data[idx]['text']
            sentiment = self.data[idx]['sentiment']
        else:
            text = self.data[idx]
            sentiment = None
        if not isinstance(text, str):
            text = ""
        text = clean(text)

        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=256,
            pad_to_max_length=True,
            truncation=True,
            # return_tensors='pt',
            return_attention_mask=True
        )

        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask'] if 'attention_mask' in encoded else None
        token_type_ids = encoded['token_type_ids'] if 'token_type_ids' in encoded else None
        
        input = {
            'input_ids':input_ids,
            'attention_mask':attention_mask,
            'token_type_ids':token_type_ids,
            'label':sentiment
        }

        for k, v in input.items():
            input[k] = torch.tensor(v) if v is not None else torch.tensor(0)
        
        return input['input_ids'], input['attention_mask'], input['token_type_ids'], input['label']

    def __len__(self):
        return len(self.data)

# TODO yaml 사용하여 파싱 짜기 / max_length, batch size, epoch 등
def build_loader(args, tokenizer, mode):
    
    if mode == 'train':
        with open(f'{args.source_data_dir}_train.json', 'r', encoding='utf-8') as f:
            source_data = json.load(f)
        with open(f'{args.target_data_dir}.json', 'r', encoding='utf-8') as f:
            target_data = json.load(f)
        random.shuffle(source_data)
        random.shuffle(target_data)
        
        train_size = int(0.9 * len(source_data))
        source_t_data = source_data[:train_size]
        source_v_data = source_data[train_size:]
        # target_data = target_data[:train_size]
        target_data = _get_target(tokenizer)
     
        source_t_dataset = BuildDataset(source_t_data, tokenizer, 'source')
        source_v_dataset = BuildDataset(source_v_data, tokenizer, 'source', 'valid')
        target_dataset = BuildDataset(target_data, tokenizer, 'target')
        
        source_t_iterator = DataLoader(dataset=source_t_dataset, sampler=RandomSampler(source_t_dataset), batch_size=args.train_batch_size)
        source_v_iterator = DataLoader(dataset=source_v_dataset, sampler=SequentialSampler(source_v_dataset), batch_size=args.eval_batch_size)
        target_iterator = DataLoader(dataset=target_dataset, sampler=SequentialSampler(target_dataset), batch_size=args.train_batch_size)

        return source_t_iterator, source_v_iterator, target_iterator
    else:
        with open(f'{args.test_data_dir}_test.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        random.shuffle(data)
        
        test_dataset = BuildDataset(data, tokenizer, 'source')
        test_iterator = DataLoader(dataset=test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.eval_batch_size)

        return test_iterator

def _get_target(tokenizer):
    with open(f'../data/sports.json', 'r', encoding='utf-8') as f:
            sports_data = json.load(f)
    with open(f"../data/tv.json", 'r', encoding='utf-8') as f:
        tv_data = json.load(f)

    sports_data = [line['text'] for line in sports_data if len(line['text']) > 5]
    tv_data = [line['text'] for line in tv_data if len(line['text']) > 5]

    _sports_data, _tv_data = [], []
    sports_total, tv_total = 0, 0
    while sports_total != 16875 or tv_total != 16875:   
        _sports_data.extend(random.sample(sports_data, 16875-sports_total))
        _tv_data.extend(random.sample(tv_data, 16875-tv_total))

        _sports_data = [line for line in _sports_data if len(tokenizer.tokenize(line)) > 2]
        _tv_data = [line for line in _tv_data if len(tokenizer.tokenize(line)) > 2]

        sports_total = len(_sports_data)
        tv_total = len(_tv_data)

    total = _sports_data + _tv_data
   
    target_data = []
    for _ in range(4):
        temp = copy.deepcopy(total)
        random.shuffle(temp)
        target_data.extend(temp)
    
    return target_data