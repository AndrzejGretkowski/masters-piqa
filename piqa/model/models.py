from transformers import (AlbertConfig, AlbertForMultipleChoice, RobertaConfig,
                          RobertaForMultipleChoice, DistilBertModel, DistilBertConfig
)

from piqa.model.model_base import BaseModelPIQA


class PIQAModel(object):
    model_mapping = dict()

    @classmethod
    def register(cls, *args):
        def decorator(fn):
            for arg in args:
                cls.model_mapping[arg] = fn
            return fn
        return decorator

    @classmethod
    def get(cls, model):
        return cls.model_mapping.get(model)


@PIQAModel.register('roberta-base', 'roberta-large')
class RobertaPIQA(BaseModelPIQA):
    @property
    def get_config(self):
        return RobertaConfig

    @property
    def get_model(self):
        return RobertaForMultipleChoice


@PIQAModel.register('albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2', 'albert-xxlarge-v2')
class AlbertPIQA(BaseModelPIQA):
    @property
    def get_config(self):
        return AlbertConfig

    @property
    def get_model(self):
        return AlbertForMultipleChoice


@PIQAModel.register('distilbert-base-uncased')
class DistilPIQA(BaseModelPIQA):
    @property
    def get_config(self):
        return DistilBertConfig

    @property
    def get_model(self):
        return DistilBertModel
