import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from transformers import AutoModel, AutoConfig

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class DomainAdaptationModel(nn.Module):
    def __init__(self, args, device):
        super(DomainAdaptationModel, self).__init__()
        self.args = args
        self.num_labels = self.args.num_labels
        self.label_lst = [0, 1]
        self.config = AutoConfig.from_pretrained(args.model_name_or_path,
                                                num_labels=self.num_labels, 
                                                finetuning_task=args.task,
                                                id2label={str(i): label for i, label in enumerate(self.label_lst)},
                                                label2id={label: i for i, label in enumerate(self.label_lst)})
        self.bert = AutoModel.from_pretrained(self.args.model_name_or_path, config=self.config)

        self.dropout = nn.Dropout(args.dropout)
        self.sentiment_classifier = nn.Linear(args.hidden_size, self.num_labels)   
        self.domain_classifier = nn.Linear(args.hidden_size, self.num_labels)
        
        self.device = device


    def forward(
          self,
          input_ids=None,
          attention_mask=None,
          token_type_ids=None,
          labels=None,
          grl_lambda = 1.0, 
          ):

        outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
        )
        
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        reversed_pooled_output = GradientReversalFn.apply(pooled_output, grl_lambda)
        # pooled_output = [batch_size, hidden_size]


        sentiment_pred = self.sentiment_classifier(pooled_output)
        domain_pred = self.domain_classifier(reversed_pooled_output)
        # pred(logits) = [batch_size, num labels]

        return sentiment_pred.to(self.device), domain_pred.to(self.device)