#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 12:06:22 2017

@author: Work
"""

import nltk
from nltk.book import *
from __future__ import print_function
from __future__ import division

def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab - english_vocab
    return sorted(unusual)

unusual_words(nltk.corpus.gutenberg.words('austen-sense.txt'))

unusual_words(nltk.corpus.nps_chat.words())

from nltk.corpus import stopwords
stopwords.words('english')
len(stopwords.words('english'))

def content_fraction(text):
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwords]
    #return content
    return len(content) / len(text)

content_fraction(nltk.corpus.reuters.words())

len(nltk.corpus.brown.words())
content_fraction(nltk.corpus.brown.words())

puzzle_letters = nltk.FreqDist('egivrvonl')
obligatory = 'r'
wordlist = nltk.corpus.words.words()
[w for w in wordlist if len(w) >= 6
                         and obligatory in w
                         and nltk.FreqDist(w) <= puzzle_letters]

names = nltk.corpus.names
names.fileids()
male_names = names.words('male.txt')
female_names = names.words('female.txt')
[w for w in male_names if w in female_names]

cfd = nltk.ConditionalFreqDist(
        (fileid, name[-1])
        for fileid in names.fileids()
        for name in names.words(fileid))
cfd.plot()

entries = nltk.corpus.cmudict.entries()
len(entries)

for entry in entries[42371:42379]:
    print(entry)

syllable = ['N', 'IHO', 'K', 'S']
[word for word, pron in entries if pron[-4:] == syllable]

[w for w, pron in entries if pron[-1] == 'M' and w[-1] == 'n']

from nltk.corpus import swadesh
swadesh.fileids()
swadesh.words('en')

fr2en = swadesh.entries(['fr', 'en'])
fr2en
translate = dict(fr2en)
translate['chien']
translate['jeter']

de2en = swadesh.entries(['de', 'en'])
es2en = swadesh.entries(['es', 'en'])
translate.update(dict(de2en))
translate.update(dict(es2en))
translate['Hund']
translate['perro']

languages = ['en', 'de', 'nl', 'es', 'fr', 'pt', 'la']
for i in [139, 140, 141, 142]:
    print(swadesh.entries(languages)[i])
    
from nltk.corpus import toolbox
toolbox.entries('rotokas.dic')

from nltk.corpus import wordnet as wn
wn.synsets('motorcar')
wn.synset('car.n.01').lemma_names()

wn.synset('car.n.01').definition()
wn.synset('car.n.01').examples()

wn.synset('car.n.01').lemmas()
wn.lemma('car.n.01.automobile')
wn.lemma('car.n.01.automobile').synset()
wn.lemma('car.n.01.automobile').name()

wn.synsets('car')
for synset in wn.synsets('car'):
    print(synset.lemma_names())
wn.lemmas('car')

motorcar = wn.synset('car.n.01')
types_of_motorcar = motorcar.hyponyms()
types_of_motorcar[0]
sorted(lemma.name() for synset in types_of_motorcar for lemma in synset.lemmas())

motorcar.hypernyms()
paths = motorcar.hypernym_paths()
len(paths)
[synset.name() for synset in paths[0]]
[synset.name() for synset in paths[1]]













