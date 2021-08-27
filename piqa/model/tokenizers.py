from transformers import AlbertTokenizerFast, RobertaTokenizerFast

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


@PIQATokenizer.register('albert-base-v2', 'albert-large-v2')
class AlbertIQATokenizer(BaseTokenizerPIQA):
    @property
    def get_tokenizer(self):
        return AlbertTokenizerFast
