import os
import sys
import logging

from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

sys.path.append(os.path.dirname(__file__))

from sentence_cleaning_classes import (RemoveAbbreviations, RemoveHtml,
                                       RemoveNumbers, RemovePatterns,
                                       RemovePunctuations, RemoveUrl)
from text_processing_functions import (lemmatize_text,
                                       remove_stopwords,
                                       stem_text)


class TextPreprocessingStep:

    def __init__(self):
        self.processing_steps = {'sentence': [RemoveNumbers(), RemoveHtml(), RemoveUrl(),
                                              RemovePunctuations(), RemovePatterns(),
                                              RemoveAbbreviations()],
                                 'tokens': [remove_stopwords, stem_text, lemmatize_text]}

    def ind_preprocess_tokens(self, text, tokenized=False):
        """
        Put everything in lowercase, remove punctuation and stopwords --> possibility to do stemming or lemmatizaion
        """
        # Tokenize the text and convert to lowercase every word
        if not isinstance(text, list):
            tokens = word_tokenize(text)
        else:
            tokens = text
        
        for processing_step in self.processing_steps['tokens']:
            tokens = processing_step(tokens)
        
        if tokenized:
            return tokens
        # Join tokens back into a single string
        return TreebankWordDetokenizer().detokenize(tokens)

    def preprocess_text_workflow(self, text_df, tokenized):
        text_df = text_df.str.lower()

        for sent_step in self.processing_steps['sentence']:
            logging.info(sent_step)
            text_df = sent_step.clean(text_df)
        
        logging.info("Applying tokens steps")
        text_df = text_df.apply(self.ind_preprocess_tokens,
                                tokenized=tokenized)
        logging.info("Finished tokens steps")
        return text_df

    def main(self, data):
        text_df = data['Text']
        processed_text_df = self.preprocess_text_workflow(text_df, tokenized=False)
        data['Text'] = processed_text_df
        return data.loc[:, ['Text']]
