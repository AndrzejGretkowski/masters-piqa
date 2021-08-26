from transformers import RobertaForMultipleChoice, RobertaConfig
from transformers.modeling_outputs import MultipleChoiceModelOutput
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class RobertaPIQA(pl.LightningModule):
    def __init__(self, learning_rate: float, roberta_type: str = 'roberta-base'):
        super().__init__()
        config = RobertaConfig.from_pretrained(roberta_type)
        config.num_labels = 2
        self.num_labels = config.num_labels
        self.config = config
        self.lr = learning_rate

        self.model = RobertaForMultipleChoice(config).from_pretrained(roberta_type, num_labels=self.num_labels)
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
        out = torch.argmax(output.logits, dim=1)
        correct = sum(out == label).item()
        acc = correct / len(label)

        # make so that the loss is summed
        self.log('val_loss', loss)
        self.log('val_accuracy', acc, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'logits': output.logits, 'output': out, 'correct': correct}

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

        out = torch.argmax(output.logits, dim=1)

        self.log(f'test_loss', 0.0)
        return {'loss': 0.0, 'logits': output.logits, 'output': out}

    def validation_epoch_end(self, outs):
        # outs is a list of whatever you returned in `validation_step`
        loss = torch.stack(outs).mean()
        self.log("val_loss", loss)

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
                lr=self.lr,
                )
        return optimizer