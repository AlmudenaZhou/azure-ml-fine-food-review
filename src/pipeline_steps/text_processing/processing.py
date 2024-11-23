import logging

from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

from src.pipeline_steps.text_processing.sentence_cleaning_classes import RemoveAbbreviations, RemoveHtml, RemoveNumbers, RemovePatterns, RemovePunctuations, RemoveUrl
from src.pipeline_steps.text_processing.text_processing_functions import lemmatize_text, remove_stopwords, stem_text


def ind_preprocess_text(text, processing_steps, tokenized=False):
    ''' Put everything in lowercase, remove punctuation and stopwords --> possibility to do stemming or lemmatizaion'''
    # Tokenize the text and convert to lowercase every word
    if not isinstance(text, list):
        tokens = word_tokenize(text)
    else:
        tokens = text
    
    for processing_step in processing_steps:
        tokens = processing_step(tokens)
    
    if tokenized:
        return tokens
    # Join tokens back into a single string
    return TreebankWordDetokenizer().detokenize(tokens)


def preprocess_text_workflow(text_df, processing_steps, tokenized):
    text_df = text_df.str.lower()

    for sent_step in processing_steps['sentence']:
        logging.info(sent_step)
        text_df = sent_step.clean(text_df)
    
    logging.info("Applying tokens steps")
    text_df = text_df.apply(ind_preprocess_text, 
                            processing_steps=processing_steps['tokens'], 
                            tokenized=tokenized)
    logging.info("Finished tokens steps")
    return text_df


def preprocess_text(text_df):
    processing_steps = {'sentence': [RemoveNumbers(), RemoveHtml(), RemoveUrl(), RemovePunctuations(), 
                                     RemovePatterns(), RemoveAbbreviations()],
                        'tokens': [remove_stopwords, stem_text, lemmatize_text]}

    processed_text_df = preprocess_text_workflow(text_df, processing_steps, tokenized=False)
    return processed_text_df
