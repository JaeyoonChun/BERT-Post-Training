import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from transformers import AutoModel, AutoConfig

class PostTrainingModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()

        self.bert = AutoModel.from_pretrained(args.bert_model).to(device)
        self.classifer = nn.Linear(args.hidden_size, 2)
        self.decoder = nn.Linear(args.hidden_size, 30000, bias=False)
        self.bias = nn.Parameter(torch.zeros(30000))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        next_sentence_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output, pooled_output = outputs[:2]
        seq_relationship_score = self.classifer(pooled_output)
        prediction_scores = self.decoder(sequence_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, 30000), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        
        output = (prediction_scores, seq_relationship_score) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output

