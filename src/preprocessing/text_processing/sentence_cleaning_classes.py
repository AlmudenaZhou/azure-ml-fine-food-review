from abc import ABC
import re


class SentenceDfCleaner(ABC):

    def __init__(self):
        self.pattern: str

    def clean(self, df):
        return df.str.replace(self.pattern, '', regex=True)


class RemoveNumbers(SentenceDfCleaner):

    def __init__(self):
        self.pattern = re.compile("\S*\d\S*")


class RemoveHtml(SentenceDfCleaner):

    def __init__(self):
        self.pattern = re.compile('<.*?>')


class RemoveUrl(SentenceDfCleaner):

    def __init__(self):
        self.pattern = re.compile('http\S+|www.\S+')


class RemovePunctuations(SentenceDfCleaner):

    def __init__(self):
        self.pattern = re.compile('[^\w\s]')


class RemovePatterns(SentenceDfCleaner):
    """
    https://stackoverflow.com/questions/37012948/regex-to-match-an-entire-word-that-contains-repeated-character
    Remove words like 'zzzzzzzzzzzzzzzzzzzzzzz', 'testtting', 'grrrrrrreeeettttt' etc. 
    Preserves words like 'looks', 'goods', 'soon' etc. We will remove all such words 
    which has three consecutive repeating characters.
    """
    def __init__(self):
        self.pattern = re.compile('\\s*\\b(?=\\w*(\\w)\\1{2,})\\w*\\b')


class RemoveAbbreviations(SentenceDfCleaner):

    def __init__(self):
        self.abbr_dict = {
            "what's":"what is",
            "what're":"what are",
            "who's":"who is",
            "who're":"who are",
            "where's":"where is",
            "where're":"where are",
            "when's":"when is",
            "when're":"when are",
            "how's":"how is",
            "how're":"how are",

            "i'm":"i am",
            "we're":"we are",
            "you're":"you are",
            "they're":"they are",
            "it's":"it is",
            "he's":"he is",
            "she's":"she is",
            "that's":"that is",
            "there's":"there is",
            "there're":"there are",

            "i've":"i have",
            "we've":"we have",
            "you've":"you have",
            "they've":"they have",
            "who've":"who have",
            "would've":"would have",
            "not've":"not have",

            "i'll":"i will",
            "we'll":"we will",
            "you'll":"you will",
            "he'll":"he will",
            "she'll":"she will",
            "it'll":"it will",
            "they'll":"they will",

            "isn't":"is not",
            "wasn't":"was not",
            "aren't":"are not",
            "weren't":"were not",
            "can't":"can not",
            "couldn't":"could not",
            "don't":"do not",
            "didn't":"did not",
            "shouldn't":"should not",
            "wouldn't":"would not",
            "doesn't":"does not",
            "haven't":"have not",
            "hasn't":"has not",
            "hadn't":"had not",
            "won't":"will not",
            '\s+':' '
        }
        self.pattern = re.compile("|".join(map(re.escape, self.abbr_dict.keys())))
    
    def clean(self, df):
        return df.str.replace(self.pattern, 
                              lambda match: self.abbr_dict[match.group(0)],
                                regex=True)
