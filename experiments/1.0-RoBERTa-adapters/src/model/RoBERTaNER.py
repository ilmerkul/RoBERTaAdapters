from transformers import RobertaModel
import torch.nn as nn

roberta_embedding_size = 768


class NERLinear(nn.Module):
    def __init__(self, lin_size, n_classes):
        super(NERLinear, self).__init__()

        self.lin0 = nn.Linear(roberta_embedding_size, lin_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=0.2)
        self.lin1 = nn.Linear(lin_size, n_classes)

    def forward(self, hidden_state):
        output = self.lin0(hidden_state)
        output = self.act(output)
        output = self.dropout(output)
        output = self.lin1(output)

        return output


class RoBERTaNER(nn.Module):
    def __init__(self, lin_size, n_classes):
        super(RoBERTaNER, self).__init__()

        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.roberta.pooler = None
        self.lin = NERLinear(lin_size, n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.roberta(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)

        output = self.lin(output["last_hidden_state"])

        return output
