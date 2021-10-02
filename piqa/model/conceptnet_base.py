from collections import OrderedDict
from re import split
from typing import List

from piqa.utils.conceptnet import ConceptNetLocalInterface
from piqa.utils.extract import WordExtractor
from piqa.utils.wiki import Wiktionary


class ConceptNetTokenizer(object):
    def __init__(self, experiment_type):
        self._type = experiment_type

        self._correct_relations = set([
            'CapableOf',
            'UsedFor',
            'Causes',
            'CausesDesire',
            'CreatedBy',
            'MotivatedByGoal',
            'ReceivesAction',
            'HasSubevent',
            'HasFirstSubevent',
            'HasLastSubevent',
            'HasPrerequisite',
            'MadeOf',
            'LocatedNear',
            'AtLocation',
            'Entails',  # Deprecated
        ])

        if self._type != "baseline":
            self.extractor = WordExtractor()
            self.wiki = Wiktionary()
            self.concept = ConceptNetLocalInterface()

    def get_important_phrases(self, text: str) -> List[str]:
        phrases = self.extractor(text)
        if len(phrases) == 0:
            phrases.extend([self.extractor.lemmatize(t) for t in text.split() if len(t) > 1][:7])
        return phrases

    def get_definition(self, phrase: str) -> List[str]:
        try:
            return self.wiki(phrase)
        except LookupError:
            return []

    def get_relations(self, phrase_1, phrase_2) -> List[str]:
        return self.concept(phrase_1, phrase_2)

    def definition_parse(self, text: str) -> str:
        phrases = self.get_important_phrases(text)
        return self._get_definition_from_phrases(phrases)[:250]

    def _get_definition_from_phrases(self, phrases: List[str]) -> str:
        for phrase in phrases:
            definitions = self.get_definition(phrase)
            if definitions:
                return f'{phrase} is {definitions[0]}'

        return ''

    def affordance_parse(self, goal: str, solution: str) -> List[str]:
        goal_phrases = list(OrderedDict.fromkeys(self.get_important_phrases(goal)))
        solution_phrases = list(OrderedDict.fromkeys(self.get_important_phrases(solution)))

        relation = None
        sol_texts = []
        for goal_phrase in goal_phrases:
            for sol_phrase in solution_phrases:
                if sol_phrase == goal_phrase:
                    continue

                relations = set(self.get_relations(goal_phrase, sol_phrase))
                common_relations = self._correct_relations & relations

                for common_relation in common_relations:
                    if relation is None:
                        relation = common_relation
                        top_goal = goal_phrase
                        top_sol = sol_phrase
                    sol_texts.append(self.concept.get_surface_text(top_goal, top_sol, relation))

        if relation is not None:
            indx = goal_phrases.index(top_goal)
            goal_phrases.insert(0, goal_phrases.pop(indx))

            indx = solution_phrases.index(top_sol)
            solution_phrases.insert(0, solution_phrases.pop(indx))

        sol_def = self._get_definition_from_phrases(solution_phrases)[:250]
        goal_def = self._get_definition_from_phrases(goal_phrases)[:250]

        sol_texts = set([s for s in sol_texts if s])
        if sol_texts:
            sol_text = '. '.join(sol_texts) + '.'
            sol_text = sol_text.replace('..', '.')
            return [sol_text, sol_def, goal_def]
        else:
            return [sol_def, goal_def]
