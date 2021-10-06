from transformers import AlbertTokenizerFast, RobertaTokenizerFast, DistilBertTokenizerFast

from piqa.model.tokenizers_base import BaseTokenizerPIQA


class PIQATokenizer(object):
    tokenizer_mapping = dict()

    @classmethod
    def register(cls, *args):
        def decorator(fn):
            for arg in args:
                cls.tokenizer_mapping[arg] = fn
            return fn
        return decorator

    @classmethod
    def get(cls, tokenizer):
        return cls.tokenizer_mapping.get(tokenizer)


@PIQATokenizer.register('roberta-base', 'roberta-large')
class RobertaPIQATokenizer(BaseTokenizerPIQA):
    @property
    def get_tokenizer(self):
        return RobertaTokenizerFast


@PIQATokenizer.register('albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2', 'albert-xxlarge-v2')
class AlbertPIQATokenizer(BaseTokenizerPIQA):
    @property
    def get_tokenizer(self):
        return AlbertTokenizerFast

@PIQATokenizer.register('distilbert-base-uncased')
class DistilPIQATokenizer(BaseTokenizerPIQA):
    @property
    def get_tokenizer(self):
        return DistilBertTokenizerFast
