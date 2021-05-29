from piqa.data.constants import LOCAL_FILE_LOCATION, PIQA_DATA_SETS

from pathlib import Path
import json


class Loader(object):
    def __init__(self, set_name='train'):
        if set_name not in PIQA_DATA_SETS:
            raise RuntimeError(f'Wrong set name, "{set_name}" does not exist.')
        self._set_name = set_name

    def load(self):
        questions_path, labels_path = PIQA_DATA_SETS[self._set_name]

        questions = self._load_questions(Path(LOCAL_FILE_LOCATION, self._set_name, questions_path))

        if labels_path is not None:
            labels = self._load_labels(Path(LOCAL_FILE_LOCATION, self._set_name, labels_path))
            if len(labels) != len(questions):
                raise RuntimeError(f'{self._set_name} set has loaded incorrectly!')
            for q, l in zip(questions, labels):
                q.update({'label': l})
    
        return questions

    @staticmethod
    def _load_labels(file_path):
        labels = []
        with open(file_path, 'rt') as f:
            for line in f:
                labels.append(int(line.strip()))
        return labels

    @staticmethod
    def _load_questions(file_path):
        questions = []
        with open(file_path, 'rt') as f:
            for line in f:
                questions.append(json.loads(line))
        return questions
