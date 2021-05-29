from transformers import RobertaTokenizerFast

import torch
from torch.nn.utils.rnn import pad_sequence



class RobertaPIQATokenizer(RobertaTokenizerFast):

    def __call__(self, sequences, max_size=150):
        return self.build_with_special_tokens(
            [super(RobertaTokenizerFast, self).__call__(seq, add_special_tokens=False) for seq in sequences], max_size=max_size)

    def build_with_special_tokens(self, embeddings, max_size):
        output = {'input_ids': []}
        for embedding in embeddings:
            if not output['input_ids']:
                output['input_ids'] += [self.bos_token_id] + embedding['input_ids'][:max_size] + [self.eos_token_id]
            else:
                output['input_ids'] += [self.eos_token_id] + embedding['input_ids'][:max_size] + [self.eos_token_id]
        output['attention_mask'] = [1] * len(output['input_ids'])
        return output

    @staticmethod
    def collate_fn(batch, pad_token, att_token = 0):
        goal, sol1, sol2, label = [], [], [], []
        input1, input2, mask1, mask2 = [], [], [], []
        for item in batch:
            goal.append(item['goal'])
            sol1.append(item['sol1'])
            sol2.append(item['sol2'])
            input1.append(torch.LongTensor(item['input1']))
            input2.append(torch.LongTensor(item['input2']))
            mask1.append(torch.LongTensor(item['mask1']))
            mask2.append(torch.LongTensor(item['mask2']))

            if 'label' in item:
                label.append(item['label'])

        return_fn = {
            'goal': goal,
            'sol1': sol1,
            'sol2': sol2,
            'input1': pad_sequence(input1, batch_first=True, padding_value=pad_token),
            'input2': pad_sequence(input2, batch_first=True, padding_value=pad_token),
            'mask1': pad_sequence(mask1, batch_first=True, padding_value=att_token),
            'mask2': pad_sequence(mask2, batch_first=True, padding_value=att_token),
        }

        if len(label) == len(goal):
            return_fn.update({'label': torch.LongTensor(label)})

        return return_fn

    def tokenize_data_set(self, dataset):
        for item in dataset:
            sol1_emb = self([item['goal'], item['sol1']])
            sol2_emb = self([item['goal'], item['sol2']])
            item.update({
                'input1': sol1_emb['input_ids'], 'mask1': sol1_emb['attention_mask'],
                'input2': sol2_emb['input_ids'], 'mask2': sol2_emb['attention_mask'],
            })
        return dataset
