from abc import ABC, abstractmethod
import pickle

import torch
from torch.nn.utils.rnn import pad_sequence

from piqa.model.conceptnet_base import AffordanceType, ConceptNetTokenizer


class BaseTokenizerPIQA(ConceptNetTokenizer, ABC):
    def __init__(self, experiment_type, ngram, return_words, definition_length, affordance_type, model_type, tqdm_arg):
        super().__init__(experiment_type, ngram, return_words, definition_length, affordance_type)

        if tqdm_arg:
        # TQDM
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        self._tqdm_func = tqdm

        self._model_type = model_type
        self.tokenizer = self.get_tokenizer.from_pretrained(model_type)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.sep_token = self.tokenizer.sep_token

    @property
    @abstractmethod
    def get_tokenizer(self):
        """Get tokenizer class."""

    def pretokenize_data_set(self, dataset):
        if self._type == "baseline":
            return self.tokenize_data_set_baseline(dataset)
        elif self._type == "definition":
            return self.tokenize_data_set_definition(dataset)
        elif self._type == "affordance":
            return self.tokenize_data_set_affordances(dataset)
        else:
            raise RuntimeError(f"Wrong type of experiment: {self._type}")

    def tokenize_data_set(self, dataset):
        for item in dataset:
            sol1 = self.append_text(item['sol1'][0], item['sol1'][1:])
            sol2 = self.append_text(item['sol2'][0], item['sol2'][1:])

            out = self.tokenizer([item['goal']] * 2, [sol1, sol2], return_tensors='pt', padding=True, truncation=True)
            out['input_ids'] = out['input_ids'].transpose(1, 0)
            out['attention_mask'] = out['attention_mask'].transpose(1, 0)
            out['token_type_ids'] = torch.zeros(out['input_ids'].size(), dtype=torch.long)
            item.update(out)
        return dataset

    def tokenize_data_set_baseline(self, dataset):
        for item in dataset:
            item.update({
                'sol1': [item['sol1']],
                'sol2': [item['sol2']],
            })

        return dataset

    def tokenize_data_set_definition(self, dataset):
        for item in self._tqdm_func(dataset):

            goal_def = self.definition_parse(item['goal'])
            sol1_def = self.definition_parse(item['sol1'])
            sol2_def = self.definition_parse(item['sol2'])

            item.update({
                'sol1': [item['sol1'], sol1_def, goal_def],
                'sol2': [item['sol2'], sol2_def, goal_def],
            })

        return dataset

    def tokenize_data_set_affordances(self, dataset):
        for item in self._tqdm_func(dataset):

            sol1_defs = self.affordance_parse(item['goal'], item['sol1'])
            sol2_defs = self.affordance_parse(item['goal'], item['sol2'])

            item.update({
                'sol1': [item['sol1']] + sol1_defs,
                'sol2': [item['sol2']] + sol2_defs,
            })

        return dataset

    @staticmethod
    def collate_fn(batch, pad_token, att_token = 0):
        goal, sol1, sol2, label = [], [], [], []
        input, mask, type = [], [], []
        for item in batch:
            goal.append(item['goal'])
            sol1.append(item['sol1'])
            sol2.append(item['sol2'])
            input.append(torch.LongTensor(item['input_ids']))
            mask.append(torch.LongTensor(item['attention_mask']))
            type.append(torch.LongTensor(item['token_type_ids']))

            if 'label' in item:
                label.append(item['label'])

        return_fn = {
            'goal': goal,
            'sol1': sol1,
            'sol2': sol2,
            'input_ids': pad_sequence(input, batch_first=True, padding_value=pad_token).transpose(1, 2).contiguous(),
            'attention_mask': pad_sequence(mask, batch_first=True, padding_value=att_token).transpose(1, 2).contiguous(),
            'token_type_ids': pad_sequence(type, batch_first=True, padding_value=0).transpose(1, 2).contiguous()
        }

        if len(label) == len(goal):
            return_fn.update({'label': torch.LongTensor(label)})

        return return_fn

    def append_text(self, text, texts_to_append):
        res_text = text

        if self._model_type.startswith('roberta'):
            for text_to_append in texts_to_append:
                if text_to_append:
                    res_text += f'{self.sep_token}{self.sep_token}{text_to_append}'
        else:
            for text_to_append in texts_to_append:
                if text_to_append:
                    res_text += f'{self.sep_token}{text_to_append}'

        return res_text
