from abc import ABC, abstractmethod

import torch
from torch.nn.utils.rnn import pad_sequence


class BaseTokenizerPIQA(ABC):
    def __init__(self, model_type):
        self.tokenizer = self.get_tokenizer.from_pretrained(model_type)
        self.pad_token_id = self.tokenizer.pad_token_id

    @property
    @abstractmethod
    def get_tokenizer(self):
        """Get tokenizer class."""

    def tokenize_data_set(self, dataset):
        for item in dataset:
            out = self.tokenizer([item['goal']] * 2, [item['sol1'], item['sol2']], return_tensors='pt', padding=True, truncation=True)
            out['input_ids'] = out['input_ids'].transpose(1, 0)
            out['attention_mask'] = out['attention_mask'].transpose(1, 0)
            out['token_type_ids'] = torch.zeros(out['input_ids'].size(), dtype=torch.long)
            item.update(out)
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
