import json
import random
import copy
import logging
import argparse
import os
import yaml
from datetime import datetime
from tqdm import tqdm, trange
from utils import set_seeds, clean
from tokenization_kobert import KoBertTokenizer
from PostTrainingModel import PostTrainingModel

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import AutoTokenizer, AutoModelForPreTraining
from transformers import get_linear_schedule_with_warmup, AdamW
from transformers import WEIGHTS_NAME, CONFIG_NAME

TIME_STAMP = datetime.now().strftime('%Y-%m-%dT%H:%M')
if not os.path.exists(f'./checkpoints/post_{TIME_STAMP}/'):
    os.mkdir(f'./checkpoints/post_{TIME_STAMP}/')
    
config = yaml.load(open('./post_logger.yml'), Loader=yaml.FullLoader)
config['handlers']['file_info']['filename'] = f'./checkpoints/post_{TIME_STAMP}/train.log'
logging.config.dictConfig(config)
logger = logging.getLogger()

class PostTraining(Dataset):
    def __init__(self, args, tokenizer):
        self.tokenizer = tokenizer
        self.sample_counter = 0
        self.max_seq_length = args.max_seq_length

        with open(f'../data/sports.json', 'r', encoding='utf-8') as f:
            sports_data = json.load(f)
        with open(f"../data/tv.json", 'r', encoding='utf-8') as f:
            tv_data = json.load(f)
        with open(f"../data/news_content_number.txt", 'r', encoding='utf-8') as f:
            mix_data = f.readlines()

        sports_data = [line['text'] for line in sports_data if len(line['text']) > 5]
        tv_data = [line['text'] for line in tv_data if len(line['text']) > 5]
        self.mix_data = [line for line in mix_data if len(line) > 10]

        _sports_data, _tv_data = [], []
        sports_total, tv_total = 0, 0
        while sports_total != 8000 or tv_total != 8000:   
            _sports_data.extend(random.sample(sports_data, 8000-sports_total))
            _tv_data.extend(random.sample(tv_data, 8000-tv_total))

            _sports_data = [line for line in _sports_data if len(self.tokenizer.tokenize(line)) > 2]
            _tv_data = [line for line in _tv_data if len(self.tokenizer.tokenize(line)) > 2]

            sports_total = len(_sports_data)
            tv_total = len(_tv_data)

        total = _sports_data + _tv_data
        print(len(total))
        self.target_data = []
        for _ in range(10):
            temp = copy.deepcopy(total)
            random.shuffle(temp)
            self.target_data.extend(temp)
    
    def __len__(self):
        return len(self.target_data)
    
    def __getitem__(self, idx):
        guid = self.sample_counter
        self.sample_counter += 1

        t1, t2, is_mix, t1_domain_label, t2_domain_label= self.random_sent(idx)

        tokens_a = self.tokenizer.tokenize(clean(t1))
        tokens_b = self.tokenizer.tokenize(clean(t2))

        example = InputExample(guid=guid, tokens_a=tokens_a, tokens_b=tokens_b, is_mix=is_mix,
                                t1_domain_label=t1_domain_label, t2_domain_label=t2_domain_label)
        
        features = convert_example_to_features(example, self.tokenizer, self.max_seq_length)
        
        tensors = (torch.tensor(features.input_ids),
                    torch.tensor(features.input_mask),
                    torch.tensor(features.segment_ids),
                    torch.tensor(features.mlm_label_ids),
                    torch.tensor(features.is_mix))

        return tensors
    
    def random_sent(self, idx):
        t1, t2 = self.get_target_line(idx)
        t1_domain_label = t2_domain_label = 'target'

        if random.random() > 0.5:
            is_mix = 0
        else:
            if random.random() > 0.5:
                t1 = self.get_other_line(idx)
                t1_domain_label = 'mix'
            else:
                t2 = self.get_other_line(idx)
                t2_domain_label = 'mix'
            is_mix = 1
        
        assert len(t1) > 0
        assert len(t2) > 0
        return t1, t2, is_mix, t1_domain_label, t2_domain_label

    def get_target_line(self, idx):
        t1 = self.target_data[idx]

        if idx == len(self.target_data) - 1:
            t2 = self.target_data[0]
        else:
            t2 = self.target_data[idx + 1]
        
        return t1, t2

    def get_other_line(self, idx):
        line = self.mix_data[idx]
        
        assert len(line) > 0
        return line

class InputExample:
    def __init__(self, 
                guid, 
                tokens_a, 
                tokens_b=None, 
                is_mix=None, 
                mlm_labels=None, 
                t1_domain_label=None, 
                t2_domain_label=None):
        
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.is_mix = is_mix # A, B sentence 가 다른 도메인에서 온 문장인지?
        self.mlm_labels = mlm_labels  # masked words for language model
        self.t1_domain_label = t1_domain_label
        self.t2_domain_label = t2_domain_label

class InputFeatures:
    def __init__(self, 
                input_ids, 
                input_mask, 
                segment_ids, 
                is_mix, 
                mlm_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_mix = is_mix
        self.mlm_label_ids = mlm_label_ids

def random_word(tokens, tokenizer, domain_label):

    output_label = []
    
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if domain_label == 'target':
            if prob < 0.15:
                prob /= 0.15 # TODO 논문에서는 15% of tokens를 masking 한다고 되어 있는데, 그중 80%만 하는건지는 모르겠다.

                if prob < 0.8:
                    tokens[i] = "[MASK]"
                elif prob < 0.9:
                    tokens[idx] = random.choice(list(tokenizer.vocab.items()))[0]

                # append current token to output (we will predict these later)
                try:
                    output_label.append(tokenizer.vocab[token])
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    output_label.append(tokenizer.vocab["[UNK]"])
                    logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-100)
        else:
            output_label.append(-100)

    return tokens, output_label


def convert_example_to_features(example, tokenizer, max_seq_length):
    tokens_a = example.tokens_a
    tokens_b = example.tokens_b
    
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    tokens_a, t1_label = random_word(tokens_a, tokenizer, example.t1_domain_label)
    tokens_b, t2_label = random_word(tokens_b, tokenizer, example.t2_domain_label)

    mlm_label_ids = ([-100] + t1_label + [-100] + t2_label + [-100])

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    try:
        assert len(tokens_b) > 0
    except:
        print(example.tokens_a)
        print(example.tokens_b)
        print(example.is_mix)
    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        mlm_label_ids.append(-100)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(mlm_label_ids) == max_seq_length

    if example.guid < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logger.info("MLM label: %s " % (mlm_label_ids))
        logger.info("Is mix domain label: %s " % (example.is_mix))
        logger.info("t1 domain label: %s " % (example.t1_domain_label))
        logger.info("t2 domain label: %s " % (example.t2_domain_label))

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             is_mix=example.is_mix,
                             mlm_label_ids=mlm_label_ids)
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def main(args):
    set_seeds()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PostTrainingModel(args, device).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    train_dataset = PostTraining(args, tokenizer)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", t_total)
    logger.info("  Num epochs = %d", args.num_train_epochs)
    logger.info("  Learning rate = %d", args.learning_rate)
    
    global_step = 0
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    model.train()
    for _ in train_iterator:
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            optimizer.zero_grad()

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, mlm_label_ids, is_mix = batch
            
            outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=mlm_label_ids, next_sentence_label=is_mix)
            loss = outputs[0]
        
            loss.backward()
            tr_loss += loss.item()
            nb_tr_steps += 1
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step() 
                model.zero_grad()
                global_step += 1

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        # Save a trained model

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    print(WEIGHTS_NAME)
    print(CONFIG_NAME)
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    # output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    # model_to_save.args.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)
    logger.info(f'train loss = {tr_loss / global_step}')
    logger.info(f'global steps = {global_step}')
    logger.info("=========== Saving fine - tuned model ===========")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    

    ## Required parameters
    parser.add_argument("--train_corpus",
                        default='../data/sports',
                        type=str,
                        help="The input train corpus. sports or tv")
    parser.add_argument("--mix_domain",
                        default='../data/news_content_number',
                        type=str,
                        help="The input train corpus. sports or tv")
    parser.add_argument("--bert_model", default='beomi/kcbert-base', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=f'./checkpoints/post_{TIME_STAMP}/',
                        type=str,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=5.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_steps", default=10000, type=int, help="Maximum steps size")
    parser.add_argument("--hidden_size", default=768, type=int, help="Hidden Vector size")

    args = parser.parse_args()

    main(args)