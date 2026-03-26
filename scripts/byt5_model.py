import torch
import torch.nn as nn
from transformers import T5EncoderModel

class ByT5Classifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.d_model, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        enc = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask,
                           return_dict=True)

        pooled = enc.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels.long())

        return {"loss": loss, "logits": logits}
