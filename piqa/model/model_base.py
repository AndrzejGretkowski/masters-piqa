from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaForMultipleChoice
from transformers.modeling_outputs import MultipleChoiceModelOutput


class BaseModelPIQA(pl.LightningModule, ABC):
    def __init__(self, learning_rate: float, model_type: str):
        super().__init__()
        config = RobertaConfig.from_pretrained(model_type)
        config = self.get_config.from_pretrained(model_type)
        config.num_labels = 2
        self.num_labels = config.num_labels
        self.config = config
        self.lr = learning_rate

        self.model = self.get_model(config).from_pretrained(model_type, num_labels=self.num_labels)
        # self.model.init_weights()

    @property    
    @abstractmethod
    def get_config(self):
        """Return a pretrained config class."""

    @property
    @abstractmethod
    def get_model(self):
        """Return a model class."""      

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
        out = torch.argmax(output.logits, dim=1)
        correct = sum(out == label).item()
        acc = correct / len(label)

        # make so that the loss is logged
        self.log('train_loss', loss)
        self.log('train_accuracy', acc, on_epoch=True)
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

        # make so that the loss is logged
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

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
        )
