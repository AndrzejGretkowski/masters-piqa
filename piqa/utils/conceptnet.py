from abc import ABC, abstractmethod
from typing import List

import conceptnet_lite
import requests
from conceptnet_lite import Label, edges_between
from peewee import DoesNotExist
from inflection import camelize
from ratelimit import limits, sleep_and_retry

from .cache import CacheDict


class BaseConceptNet(ABC):
    # def __init__(self, language='en', cache_size=1000):
    #     self._lang = language
    #     self._cache = CacheDict(maxlen=cache_size)

    @abstractmethod
    def get_relations_between(self, phrase_1, phrase_2):
        pass

    @abstractmethod
    def get_surface_text(self, phrase_1, phrase2, relation):
        pass

    @staticmethod
    def _join_phrase_correctly(phrase: str) -> str:
        return '_'.join(phrase.split())

    def __call__(self, phrase_1, phrase_2) -> List[str]:
        return self.get_relations_between(phrase_1, phrase_2)

    # @staticmethod
    # def _cache_wrapper(func):
    #     def wrap(word_or_phrase: str):


class ConceptNetWebInterface(BaseConceptNet):
    def __init__(self, language='en', cache_size=20000):
        self._main_url = f'https://api.conceptnet.io/query?'
        self._lang = language
        self._cache = CacheDict(maxlen=cache_size)

    def _make_concept(self, correct_phrase):
        return f'/c/{self._lang}/{correct_phrase}'

    def _get_node(self, phrase_1, phrase_2):
        correct_uri_1 = self._join_phrase_correctly(phrase_1)
        correct_uri_2 = self._join_phrase_correctly(phrase_2)

        if (correct_uri_1, correct_uri_2) in self._cache:
            return self._cache[(correct_uri_1, correct_uri_2)]
        else:
            url = f'{self._main_url}start={self._make_concept(correct_uri_1)}&end={self._make_concept(correct_uri_2)}'
            node = self._request_url(url).json()
            self._cache[(correct_uri_1, correct_uri_2)] = node
            return node

    @sleep_and_retry
    @limits(calls=120, period=120)
    def _request_url(self, url):
        return requests.get(url)

    def get_relations_between(self, phrase_1, phrase_2):
        relations = []
        edges = self._get_node(phrase_1, phrase_2)['edges']

        for edge in edges:
            relations.append(camelize(edge['rel']['label']))

        return relations

class ConceptNetLocalInterface(BaseConceptNet):
    def __init__(self, language='en', cache_size=1000, path: str = './data/conceptnet/concpetnet.db'):
        self._lang = language
        self._cache = CacheDict(maxlen=cache_size)
        self._initialized = False
        self._path = path

    def _init(self):
        if not self._initialized:
            self._connection = conceptnet_lite.connect(self._path)
            self._initialized = True

    def _get_node(self, word_or_phrase: str):
        self._init()
        correct_uri = self._join_phrase_correctly(word_or_phrase)
        if correct_uri in self._cache:
            return self._cache[correct_uri]
        else:
            node = Label.get(text=correct_uri, language=self._lang)
            self._cache[correct_uri] = node
            return node

    def get_relations_between(self, phrase_1, phrase_2):
        relations = []

        try:
            concept_1 = self._get_node(phrase_1).concepts
            concept_2 = self._get_node(phrase_2).concepts

        except DoesNotExist:
            return relations

        for e in edges_between(concept_1, concept_2, two_way=False):
            relations.append(camelize(e.relation.name))

        return relations

    def get_surface_text(self, phrase_1, phrase_2, relation):
        try:
            node1 = self._get_node(phrase_1).concepts
            node2 = self._get_node(phrase_2).concepts
        except DoesNotExist:
            return ''

        for e in edges_between(node1, node2, two_way=False):
            if camelize(e.relation.name) == relation:
                text: str = e.etc['surfaceText']
                if text:
                    return text.replace('[[', '').replace(']]', '').replace('*', '')
                return ''
