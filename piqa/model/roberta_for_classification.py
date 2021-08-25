from transformers import RobertaForMultipleChoice, RobertaConfig
from transformers.modeling_outputs import MultipleChoiceModelOutput
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class RobertaPIQA(pl.LightningModule):
    def __init__(self):
        super().__init__()
        config = RobertaConfig.from_pretrained('roberta-large')
        config.num_labels = 2
        self.num_labels = config.num_labels
        self.config = config

        self.model = RobertaForMultipleChoice(config).from_pretrained('roberta-large', num_labels=self.num_labels)
        # self.model.init_weights()

    def forward(self, *args, **kwargs) -> MultipleChoiceModelOutput:
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        # unpack batch
        input = batch['input_ids']
        mask = batch['attention_mask']
        token_type = batch['token_type_ids']
        label = batch['label']
        # forward + loss
        output = self.model(
            input_ids=input,
            attention_mask=mask,
            token_type_ids=token_type,
            labels=label)

        loss = output.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # unpack batch
        input = batch['input_ids']
        mask = batch['attention_mask']
        token_type = batch['token_type_ids']
        label = batch['label']
        # forward + loss
        output = self.model(
            input_ids=input,
            attention_mask=mask,
            token_type_ids=token_type,
            labels=label)

        loss = output.loss
        # make so that the loss is summed
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        # unpack batch
        input = batch['input_ids']
        mask = batch['attention_mask']
        token_type = batch['token_type_ids']
        # forward + loss
        output = self.model(
            input_ids=input,
            attention_mask=mask,
            token_type_ids=token_type)

        self.log(f'test_loss', 0.0)
        return {'loss': 0.0, 'logits': output.logits}

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    "weight_decay_rate": 0.01
                    },
                {
                    "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    "weight_decay_rate": 0.0
                    },
                ]
        optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=2e-5,
                )
        return optimizer