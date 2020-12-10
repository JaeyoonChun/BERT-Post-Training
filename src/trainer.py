import os
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoConfig, BertForSequenceClassification
from DANN_model import DomainAdaptationModel
from data_loader import build_loader

from utils import binary_accuracy, format_time

logger = logging.getLogger()

class Trainer(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.label_lst = [0, 1]
        self.num_labels = len(self.label_lst)
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

        self.model = DomainAdaptationModel(args, self.device)
        # GPU or CPU
        
        def truncated_normal_(tensor, mean=0.0, std=0.02):
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor
        
        # weight.data = [num labels, hidden size]
        self.model.sentiment_classifier.weight.data = nn.Parameter(truncated_normal_(self.model.sentiment_classifier.weight.data))
        self.model.domain_classifier.weight.data = nn.Parameter(truncated_normal_(self.model.domain_classifier.weight.data))
        print(self.model.sentiment_classifier.weight.data)
        self.model.to(self.device)

    def train(self):
        # TODO args fpath
        source_t_loader, source_v_loader, target_loader = build_loader(self.args, self.tokenizer, 'train')

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(source_t_loader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(source_t_loader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        loss_fn_sentiment_classifier = torch.nn.CrossEntropyLoss()
        loss_fn_domain_classifier = torch.nn.CrossEntropyLoss()
        max_batches = min(len(source_t_loader), len(target_loader))

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.00}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        best_valid_loss = float('inf')

        self.model.zero_grad()

        for epoch_idx in range(int(self.args.num_train_epochs)):
            logger.info(f"========== {epoch_idx + 1} : {self.args.num_train_epochs} ==========")
            
            epoch_train_loss, epoch_valid_loss = 0, 0
            epoch_valid_accuracy, valid_cnt = 0, 0
            
            source_iterator = iter(source_t_loader)
            target_iterator = iter(target_loader)

            for step in trange(max_batches, desc="Iteration"):
                self.model.train()
                optimizer.zero_grad()

                # calculating training progress p, adaptation rate lambda
                p = float(step + epoch_idx * max_batches) / (self.args.num_train_epochs * max_batches)
                grl_lambda = 2. / (1. + np.exp(-10 * p)) - 1
                grl_lambda = torch.tensor(grl_lambda)

                # Source domain training
                input_ids, attention_mask, token_type_ids, labels = next(source_iterator)
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids" : token_type_ids,
                    "labels" : labels,
                    "grl_lambda" : grl_lambda,
                }

                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                
                # Take both sentiment pred and domain pred for the source domain
                sentiment_pred, domain_pred = self.model(**inputs)
                # input['labels'] = [batch_size, ]
                loss_s_sentiment = loss_fn_sentiment_classifier(sentiment_pred, inputs["labels"])
                y_s_domain = torch.zeros(self.args.train_batch_size, dtype=torch.long).to(self.device)
                loss_s_domain = loss_fn_domain_classifier(domain_pred, y_s_domain)

                # Target domain training
                input_ids, attention_mask, token_type_ids, labels = next(target_iterator)
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids" : token_type_ids,
                    "labels" : labels,
                    "grl_lambda" : grl_lambda,
                }

                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
            
                _, domain_pred = self.model(**inputs)
                # Note that we are not using the sentiment predictions here for updating the weights
                y_t_domain = torch.ones(self.args.train_batch_size, dtype=torch.long).to(self.device)
                loss_t_domain = loss_fn_domain_classifier(domain_pred, y_t_domain)

                # Combining the loss 
                loss = loss_s_sentiment + loss_s_domain + loss_t_domain

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                
                loss.backward()

                epoch_train_loss += loss.item()
        
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1
                    
                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        valid_loss, valid_accuracy = self.evaluate(source_v_loader, "valid")
                        epoch_valid_loss += valid_loss
                        epoch_valid_accuracy += valid_accuracy
                        valid_cnt += 1
                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        if valid_loss < best_valid_loss:
                            self.save_model(optimizer, valid_loss)
            
            epoch_train_loss = epoch_train_loss / (step + 1)
            epoch_valid_loss = epoch_valid_loss / valid_cnt
            epoch_valid_accuracy = epoch_valid_accuracy / valid_cnt
            if epoch_valid_loss < best_valid_loss:
                best_valid_loss = epoch_valid_loss    
                self.save_model(optimizer, best_valid_loss)
            logger.info("  %s : %s |  %s = %s", 'EPOCH', epoch_idx + 1, 'train_loss', epoch_train_loss)
            logger.info("  %s : %s |  %s = %s", 'EPOCH', epoch_idx + 1, 'valid_loss', epoch_valid_loss)
            logger.info("  %s : %s |  %s = %s", 'EPOCH', epoch_idx + 1, 'valid_accuracy', epoch_valid_accuracy)

    def evaluate(self, eval_dataloader, mode):

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(eval_dataloader))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        loss_fn_sentiment_classifier = torch.nn.CrossEntropyLoss()
        preds, total_labels = [], []
        
        self.model.eval()

        eval_dataloader = iter(eval_dataloader)
        for step in trange(len(eval_dataloader), desc="Evaluating"):
            input_ids, attention_mask, token_type_ids, labels = next(eval_dataloader)
            with torch.no_grad():
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids" : token_type_ids,
                    "labels": labels
                }
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)

                sentiment_pred, _ = self.model(**inputs)
                loss_s_sentiment = loss_fn_sentiment_classifier(sentiment_pred, inputs["labels"])
                eval_loss += loss_s_sentiment.item()
                
                preds.append(sentiment_pred)
                total_labels.append(inputs['labels'])
            nb_eval_steps += 1
        
        preds = torch.cat(preds, dim=0)
        total_labels = torch.cat(total_labels)
        
        eval_loss = eval_loss / nb_eval_steps
        _, eval_accuracy = binary_accuracy(preds, total_labels)

        logger.info("  %s = %s", 'eval_loss', eval_loss)
        logger.info("  %s = %s", 'mean_accuracy', eval_accuracy)

        return eval_loss, eval_accuracy

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(f'{logger.handlers[1].baseFilename[:-10]}'):
            os.makedirs(f'{logger.handlers[1].baseFilename[:-10]}')
        torch.save(self.model.state_dict(), f'{logger.handlers[1].baseFilename[:-10]}/best_model.pt')

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(f'{logger.handlers[1].baseFilename[:-10]}', 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", f'{logger.handlers[1].baseFilename[:-10]}')

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(self.args.model_dir)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
