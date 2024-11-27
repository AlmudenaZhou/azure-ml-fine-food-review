import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


def stem_text(tokens):
    stemmer = SnowballStemmer('english')
    try:
        tokens = [stemmer.stem(word) for word in tokens]
    except TypeError:
        print(tokens)
    return tokens

def lemmatize_text(tokens):
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


def remove_stopwords(tokens):
    default_stopwords = set(stopwords.words('english'))
    excluding = set(['against','not','don', "don't",'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
             'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 
             'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",'shouldn', "shouldn't", 'wasn',
             "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])
 
    custom_stopwords = default_stopwords - excluding

    tokens = [token for token in tokens if token not in custom_stopwords]
    tokens = filter(None, tokens)
    return tokens
