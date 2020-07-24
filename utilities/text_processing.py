# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 10:47:48 2017
For textpreperocessing
@author: anthonyta
"""
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet


def get_wordnet_pos(treebank_tag):
    """Convert POS nltk treebank tag to wordnet definition

    Parameters
    ----------
    treebank_tag: a tag provided by nltk treebank pos_tag, e.g:VB, NN,...

    Returns
    -------
    a tag in WordNet definition
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    elif treebank_tag.startswith('S'):
        return wordnet.ADJ
    else:
        return wordnet.NOUN


def lemmatize_tagged_words(tagged_words):
    """Lemmatizing words based on their POS tag

    Parameters
    ----------
    tagged_words: a list of (word,tag) pairs. Tags follow WordNet definition

    Returns
    -------
    a list of lemmatized words
    """
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmed_words = []
    for token in tagged_words:
        tokenStr = str(token[1])
        word_pos = get_wordnet_pos(tokenStr)
        lemmed_words.append(wordnet_lemmatizer.lemmatize(token[0], word_pos))

    return lemmed_words


def lemmatize_words(words):
    """Lemmatizing words

    Parameters
    ----------
    words: a list of words

    Returns
    -------
    a list of lemmatized words
    """
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmed_words = []
    for word in words:
        lemmed_words.append(wordnet_lemmatizer.lemmatize(word))

    return lemmed_words


def url_removal(text):
    """Remove all forms of url from text
    Some forms of urls: https:// http:// www.

    Parameters
    ----------
    text: a string with/without urls

    Returns
    -------
    a string with urls removed
    """
    return re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]\
    {2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]\
    +|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', text)


def non_letter_removal(text):
    """Remove non-letter from text

    Parameters
    ----------
    text: a string

    Returns
    -------
    a string with non-letters removed
    """
    return re.sub('[^a-zA-Z]', ' ', text)


def stopword_removal_from_taggedwords(tagged_words):
    """Remove stopword from list of tagged words

    Parameters
    ----------
    tagged_words: a list of (word,tag) pairs. Tags follow WordNet definition

    Returns
    -------
    a list of (word,tag) pairs with every word=stopword removed
    """
    stops = set(stopwords.words('english'))
    tagged_words = [w for w in tagged_words if not w[0] in stops]
    return tagged_words


def stopword_removal(words):
    """Remove stopword from list of words
    Parameters
    ----------
    words: a list of words
    Returns
    -------
    a list of words word=stopword removed
    """
    stops = set(stopwords.words('english'))
    words = [w for w in words if w not in stops]
    return words


def text_to_wordlist(text, remove_html_related=True, remove_non_letter=True,
                     to_lowercase=True, remove_stopwords=False, use_lem=False):
    """Function to convert a text document to a sequence of words.
    1. [Optional] remove urls and html markups
    2. [Optional] remove non-letter characters
    3. [Optional] convert text to lowercase
    4. Split text into list of words
    5. Tag words with Part of Speech (POS)
    6. [Optional] remove stopwords
    7. [Optional] lemmatize the words

    Parameters
    ----------
    text: a string that need to be processed
    remove_html_related: option to remove urls and html markups,\
    default is True
    remove_non_letter: option to remove non-letter characters, default is True
    to_lowercase: option to convert text to lowercase, default is True
    remove_stopwords: option to remove stopwords, default is False
    use_lem: option to lemmatize the words using pos tags, default is False

    Returns
    -------
    a list of words (after step 7),\
    and a list of tagged words (word,tag) (after step 5)
    """
    if remove_html_related:
        text = url_removal(text)
        # Remove HTML using BeautifulSoup
        text = BeautifulSoup(text, 'lxml').get_text()

    # Remove non-letters using regex
    if remove_non_letter:
        text = non_letter_removal(text)
    # Convert words to lower case and split them
    if to_lowercase:
        text = text.lower()

    words = text.split()
    # get tagged before possible stopword removal
    tagged_words = pos_tag(words)

    # Optionally remove stop words (false by default)
    if remove_stopwords:
        tagged_words = stopword_removal_from_taggedwords(tagged_words)

    # Optionally get part of speech tag of words then lemmatize them
    if use_lem:
        words = lemmatize_tagged_words(tagged_words)
    # Return a list of words and tagged words
    return(words, tagged_words)


def text_to_sentences(text, tokenizer, remove_html_related=True,
                      remove_non_letter=True, to_lowercase=True,
                      remove_stopwords=False, use_lem=False):
    # Function to split a text into parsed sentences.
    # Returns a list of sentences, where each sentence is a list of words
    # Need to remove HTML here for better sentence splitting
    """Function to split a text into parsed sentences
    1. [Optional] remove urls and html markups
    2. Use the NLTK tokenizer to split the paragraph into sentences
    3. For each sentence, call text_to_wordlist

    Parameters
    ----------
    text: a string that need to be processed
    tokenizer: nltk tokenizer
    remove_html_related: option to remove urls and html markups,\
    default is True
    remove_non_letter: option to remove non-letter characters, default is True
    to_lowercase: option to convert text to lowercase, default is True
    remove_stopwords: option to remove stopwords, default is False
    use_lem: option to lemmatize the words using pos tags, default is False

    Returns
    -------
    a list of sentences and a list of tagged words (word,tag)
    """
    if remove_html_related:
        text = url_removal(text)
        # Remove HTML using BeautifulSoup
        text = BeautifulSoup(text, 'lxml').get_text()
    remove_html_related = False

    # Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(text.strip())

    # Loop over each sentence
    sentences = []
    tagged_words = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call text_to_wordlist to get a list of words
            (wordlist, taglist) = text_to_wordlist(raw_sentence,
                                                   remove_html_related,
                                                   remove_non_letter,
                                                   to_lowercase,
                                                   remove_stopwords,
                                                   use_lem)
            if len(wordlist) > 0:
                sentences.append(wordlist)
                tagged_words.append(taglist)

    # Return the list of sentences and tagged words
    return(sentences, tagged_words)


def text_to_words(text, remove_html_related=True, remove_non_letter=True,
                  to_lowercase=True, remove_stopwords=False, use_lem=False):
    """Function to cparse a text to a string of words
    Simply call text_to_wordlist then join all the words to a string

    Parameters
    ----------
    text: a string that need to be processed
    remove_html_related: option to remove urls and html markups,\
    default is True
    remove_non_letter: option to remove non-letter characters, default is True
    to_lowercase: option to convert text to lowercase, default is True
    remove_stopwords: option to remove stopwords, default is False
    use_lem: option to lemmatize the words using pos tags, default is False

    Returns
    -------
    a single processed string
    """
    (words, tagged_words) = text_to_wordlist(text,
                                             remove_html_related,
                                             remove_non_letter,
                                             to_lowercase,
                                             remove_stopwords,
                                             use_lem)
    return(" ".join(words), tagged_words)
