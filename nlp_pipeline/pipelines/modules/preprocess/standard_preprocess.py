import unicodedata

import numpy as np
from underthesea import ner, pos_tag, word_tokenize

import regex
from questgen.utils.constants import STOPWORD_PATH


viet_stopwords = np.load(STOPWORD_PATH)


def unicode_normalize(string, form="NFKC"):
    return unicodedata.normalize(form, string)


def word_segment(string, format="text"):
    return word_tokenize(string, format=format)


def sentence_segment(text):
    sents = regex.split("([.?!])?[\n]+|[.?!] ", text)
    return sents


def remove_hle(string):
    # Common
    string = regex.sub("(?s)<ref>.+?</ref>", "", string)  # remove reference links
    string = regex.sub("(?s)<[^>]+>", "", string)  # remove html tags
    string = regex.sub("&[a-z]+;", "", string)  # remove html entities
    string = regex.sub("(?s){{.+?}}", "", string)  # remove markup tags
    string = regex.sub("(?s){.+?}", "", string)  # remove markup tags
    string = regex.sub("(?s)\[\[([^]]+\|)", "", string)  # remove link target strings
    string = regex.sub("(?s)\[\[([^]]+\:.+?]])", "", string)  # remove media links

    string = regex.sub("[']{5}", "", string)  # remove italic+bold symbols
    string = regex.sub("[']{3}", "", string)  # remove bold symbols
    string = regex.sub("[']{2}", "", string)  # remove italic symbols

    # Common
    string = regex.sub("[ ]{2,}", " ", string)  # Squeeze spaces.
    return string


def remove_stopwords(sentence, stopwords_path=None):
    if ".npy" in stopwords_path:
        stopwords = list(np.load(stopwords_path))
    elif ".txt" in stopwords_path:
        with open(stopwords_path) as f:
            stopwords = f.read().splitlines()
    else:
        raise NotImplementedError("only support text and numpy extension")

    tokens = sentence.split(" ")
    tokens_filtered = [word for word in tokens if word not in stopwords]
    return (" ").join(tokens_filtered)


def extract_keywords(sentence: str) -> list:
    """
    Extracts keywords from the given sentence using POS tagging and NER.

    Args:
        sentence (str): The input sentence from which to extract keywords.

    Returns:
        list: A list of keywords extracted from the sentence.

    """
    entity_symbols = ("N", "Nc", "Ny", "Np", "Nu", "P", "M")

    # POS tagging
    postag_tokens = pos_tag(sentence)
    postag_tokens = [
        (word, pos) for word, pos in postag_tokens if word not in viet_stopwords
    ]
    postag_tokens = [word for word, pos in postag_tokens if pos in entity_symbols]

    # NER
    ner_tag_tokens = ner(sentence)
    ner_tag_tokens = [
        (word, begin, inside, outside)
        for word, begin, inside, outside in ner_tag_tokens
        if word not in viet_stopwords
    ]
    ner_tag_tokens = [
        word
        for word, begin, inside, outside in ner_tag_tokens
        if begin in entity_symbols
    ]
    ner_tag_tokens = [
        word
        for word, begin, inside, outside in ner_tag_tokens
        if begin in entity_symbols
    ]

    keywords = list(set(postag_tokens + ner_tag_tokens))
    return keywords
