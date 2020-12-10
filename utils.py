import os
import logging
import logging.config
import yaml
from datetime import datetime
import torch
import random
from transformers import AutoTokenizer

TIME_STAMP = datetime.now().strftime('%Y-%m-%dT%H:%M')

import re
import emoji
from soynlp.normalizer import repeat_normalize

emojis = ''.join(emoji.UNICODE_EMOJI.keys())
pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
url_pattern = re.compile(
    r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

def clean(x):
    x = pattern.sub(' ', x)
    x = url_pattern.sub('', x)
    x = x.strip()
    x = repeat_normalize(x, num_repeats=2)
    return x

def load_tokenizer(args):
    return AutoTokenizer.from_pretrained(args.model_name_or_path)

def init_logger():
    if not os.path.exists(f'post-training/checkpoints/'):
        os.mkdir(f'post-training/checkpoints/')

    if not os.path.exists(f'post-training/checkpoints/{TIME_STAMP}/'):
        os.mkdir(f'post-training/checkpoints/{TIME_STAMP}/')
    
    config = yaml.load(open('./logger.yml'), Loader=yaml.FullLoader)
    config['handlers']['file_info']['filename'] = f'post-training/checkpoints/{TIME_STAMP}/train.log'
    logging.config.dictConfig(config)

def set_seeds():
    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds)).argmax(axis=1)
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return rounded_preds, acc

def format_time(end, start):
    elapsed_time = end - start
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs