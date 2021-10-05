import nltk
import RAKE
from rake_nltk import Rake
from nltk.stem import WordNetLemmatizer
from yake import KeywordExtractor


class WordExtractor(object):
    def __init__(self, ngram: int = 1, return_words: int = 7):
        nltk.download('wordnet')

        self._lemma = WordNetLemmatizer()
        self._initalized = False
        self.ngram = ngram
        self.return_words = return_words

    def _init(self):
        if not self._initalized:
            self._extractor = KeywordExtractor(
                n=self.ngram,
                top=self.return_words
            )
            self._initalized = True

    def __call__(self, text: str) -> list:
        return self._get_ranked_phrases(text)

    def _get_ranked_phrases(self, text: str) -> list:
        self._init()
        keywords = self._extractor.extract_keywords(text)
        ret_keywords = []
        for keyword, score in keywords:
            if len(keyword.split()) > 1:
                ret_keywords.append(keyword.lower())
            else:
                ret_keywords.append(self.lemmatize(keyword.lower()))
        return ret_keywords

    def lemmatize(self, word: str) -> str:
        return self._lemma.lemmatize(word)


class WordExtractorOld(object):
    def __init__(self, use_smart_stop_list: bool = False):
        nltk.download('stopwords')
        nltk.download('wordnet')

        if use_smart_stop_list:
            self._extractor = Rake(set(RAKE.SmartStopList()))
        else:
            self._extractor = Rake()

        self._lemma = WordNetLemmatizer()

    def __call__(self, text: str, return_words: int = 5) -> list:
        phrases_with_score = self._get_ranked_phrases(text)
        phrases_with_score.sort(key=lambda x: x[0], reverse=True)
        phrases_with_score = phrases_with_score[:return_words]
        return [w for _, w in phrases_with_score]

    def _get_ranked_phrases(self, text: str) -> list:
        self._extractor.extract_keywords_from_text(text)
        return self._extractor.get_ranked_phrases_with_scores()

    def lemmatize(self, words: list) -> list:
        return [self._lemma.lemmatize(word) for word in words]