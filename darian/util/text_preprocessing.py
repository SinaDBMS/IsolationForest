import pandas as pd
import re
from typing import Set, Tuple
from collections import OrderedDict
from nltk import everygrams


def extract_n_grams(text: str, splitter: str = "\s", lowercase: bool = True,
                    ngram_range: Tuple = (1, 1)) -> Set:
    if lowercase:
        text = text.lower()

    tokens = re.split(rf"[{splitter}]", text)
    tokens = [token for token in tokens if token != '']
    tokens = list(OrderedDict.fromkeys(tokens))

    n_grams = set()

    ng = list(everygrams(tokens, ngram_range[0], ngram_range[1]))
    for n in ng:
        n_grams.add(' '.join([tuple_ for tuple_ in n]))

    return n_grams


def extract_n_grams_using_vectorizer(text, vectorizer):
    f = pd.Series(vectorizer.transform([text]).toarray()[0], index=vectorizer.get_feature_names())
    return f[f > 0].index.tolist()
