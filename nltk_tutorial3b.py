#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 12:06:22 2017

@author: Work
"""

import nltk, re, pprint
from nltk.book import *
from __future__ import print_function
from __future__ import division
from nltk import word_tokenize

raw = """DENNIS: Listen, strange women lying in ponds distributing swords 
is no basis for a system of government.  Supreme executive power derives from
a mandate from the masses, not from some farcical aquatic ceremony."""

tokens = word_tokenize(raw)

porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
[porter.stem(t) for t in tokens]
[lancaster.stem(t) for t in tokens]

class IndexedText(object):
    
    def __init__(self, stemmer, text):
        self._text = text
        self._stemmer = stemmer
        self._index = nltk.Index((self._stem(word), i)
                                for (i, word) in enumerate(text))
    
    def concordance(self, word, width=40):
        key = self._stem(word)
        wc = int(width/4)   # words of context
        for i in self._index[key]:
            lcontext = ' '.join(self._text[i-wc:i])
            rcontext = ' '.join(self._text[i:i+wc])
            ldisplay = '{:>{width}}'.format(lcontext[-width:], width=width)
            rdisplay = '{:{width}}'.format(rcontext[:width], width=width)
            print(ldisplay, rdisplay)
    
    def _stem(self, word):
        return self._stemmer.stem(word).lower()
    
grail = nltk.corpus.webtext.words('grail.txt')
text = IndexedText(porter, grail)
text.concordance('lie')

wnl = nltk.WordNetLemmatizer()
[wnl.lemmatize(t) for t in tokens]

raw = """'When I'M a Duchess,' she said to herself, (not in a very hopeful tone
though), 'I won't have any pepper in my kitchen AT ALL. Soup does very
well without--Maybe it's always pepper that makes people hot-tempered,'..."""

import re
re.split(r' ', raw)
re.split(r'[ \t\n]+', raw)
re.split(r'\s+', raw)
re.split(r'\W+', raw)
re.findall(r'\w+|\S\w*', raw)

text = 'That U.S.A. poster-print costs $12.40...'
pattern = r'''(?x)([A-Z]\.)+| \w+(-\w+)*| \$?\d+(\.\d+)?%?| \.\.\.| [][.,;"'?():-_`]'''
        
nltk.regexp_tokenize(text, pattern)

fdist = nltk.FreqDist(['dog', 'cat', 'dog', 'cat', 'dog', 'snake', 'dog', 'cat'])
for word in sorted(fdist):
    print(word, ":", fdist[word], end='; ')

# Writing results to a File
output_file = open('output.txt', 'w')
words = set(nltk.corpus.genesis.words('english-kjv.txt'))
for word in sorted(words):
    print(word, file=output_file)
len(words)
str(len(words))
print(str(len(words)), file=output_file)


