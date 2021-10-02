from pywiktionary import PageNotFoundException, WiktionaryParserFactory

from piqa.utils.cache import CacheDict


class Wiktionary(object):
    def __init__(self, language='en', maxlen=100000):
        self.parser_factory = WiktionaryParserFactory(default_language=language)
        self._cache = CacheDict(max_len=100000)

    def __call__(self, word: str, max_meanings=1) -> list:
        try:
            ret_meanings = []
            page = self.get_page(word)
            all_meanings = page['meanings']

            if 'verb' in all_meanings:
                for meaning in all_meanings['verb']['meanings']:
                    ret_meanings.append(meaning['meaning'])

            if 'noun' in all_meanings:
                for meaning in all_meanings['noun']['meanings']:
                    ret_meanings.append(meaning['meaning'])

            for key in all_meanings.keys():
                if key == 'noun' or key == 'verb':
                    continue

                for meaning in all_meanings[key]['meanings']:
                    ret_meanings.append(meaning['meaning'])

            return ret_meanings[:max_meanings]

        except PageNotFoundException:
            raise LookupError(f'Page for word {word} not found.')

    def get_page(self, word: str) -> dict:
        if word in self._cache:
            return self._cache[word]
        else:
            page = self.parser_factory.get_page(word).parse()
            self._cache[word] = page
            return page
