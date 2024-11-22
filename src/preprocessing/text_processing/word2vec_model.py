from operator import add

import pandas as pd

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize


class Word2VecModel:

    def __init__(self, min_count=1, window=10, vector_size=300, sample=6e-5, alpha=0.03,
                 min_alpha=0.0007, negative=20):

        self.model = Word2Vec(min_count=min_count,
                              window=window,
                              vector_size=vector_size,
                              sample=sample,
                              alpha=alpha,
                              min_alpha=min_alpha,
                              negative=negative)

    def _word2vec_get_vector(self, word):
        try:
            word = self.model.wv.get_vector(word)
        except KeyError:
            word = [0] * self.model.wv.vector_size
        return word


    def _ind_sentence_process(self, sentence):
        sent_vec = self._word2vec_get_vector(sentence[0])
        for word in sentence[1:]:
            sent_vec = list(map(add, sent_vec, self._word2vec_get_vector(word)))
        return list(map(lambda x: x / len(sentence), sent_vec))


    def _text_to_vec(self, x):
        x_wv = x.apply(word_tokenize).apply(self._ind_sentence_process)
        x_wv = pd.DataFrame(x_wv.tolist())
        return x_wv
    
    def fit(self, x: pd.Series, progress_per=100, epochs=30, report_delay=1):
        sentences = x.apply(word_tokenize)
        self.model.build_vocab(sentences, progress_per=progress_per)
        self.model.train(sentences, total_examples=self.model.corpus_count, epochs=epochs, report_delay=report_delay)

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = Word2Vec.load(model_path)

    def transform(self, x: pd.Series):
        data_wv = self._text_to_vec(x)
        data_wv.index = x.index
        return data_wv

    def fit_transform(self, x, **kwags):
        self.fit(x, **kwags)
        return self.transform(x)
