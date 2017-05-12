#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:54:44 2017

@author: Work
"""

from nltk.book import *
from __future__ import division
from __future__ import print_function
 
text1.concordance('monstrous')

text2.concordance('affection')
text3.concordance('lived')

text1.similar('monstrous')
text2.similar("monstrous")

text2.common_contexts(["monstrous", "very"])

text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])

len(text3)
len(text2)

sorted(set(text3))

len(set(text3))

len(set(text3)) / len(text3)

text5.count('lol')
100 * text5.count('lol') / len(text5)

def lexical_diversity(text):
    return len(set(text)) / len(text)

def percentage(count, total):
    return 100 * count / total

lexical_diversity(text3)

lexical_diversity(text5)

percentage(4,5)
percentage(text4.count('a'), len(text4))

sent1.append("Some")
sent1

text4[173]
text4.index('awaken')

saying = ['After', 'all', 'is', 'said', 'and', 'done', 'more', 'is', 'said', 'than', 'done']

tokens = set(saying)
tokens = sorted(tokens)
tokens[-2:]

fdist1 = FreqDist(text1)
print fdist1

fdist1.most_common(50)
fdist1['whale']

fdist2 = FreqDist(text2)
fdist2.most_common(50)
fdist2['love']

fdist1.hapaxes()

V = set(text1)
long_words = [w for w in V if len(w) > 15]
sorted(long_words)

fdist5 = FreqDist(text5)
sorted(w for w in set(text5) if len(w) > 7 and fdist5[w] > 7)

bigrams(['more', 'is', 'said', 'than', 'done'])

text4.collocations()
text8.collocations()

fdist = FreqDist(len(w) for w in text1)

fdist.most_common()
fdist.max()
fdist[3]
fdist.freq(3)

[w for w in sent7 if len(w) != 4]

sorted(w for w in set(text1) if w.endswith('ableness'))
sorted(term for term in set(text4) if 'gnt' in term)
sorted(item for item in set(text6) if item.istitle())
sorted(item for item in set(sent7) if item.isdigit())

sorted(w for w in set(text7) if '-' in w and 'index' in w)
sorted(wd for wd in set(text3) if wd.istitle() and len(wd) >10)
sorted(w for w in set(sent7) if not w.islower())
sorted(t for t in set(text2) if 'cie' in t or 'cei' in t)

[w.upper() for w in text1]

len(set(word.lower() for word in text1))
len(set(word.lower() for word in text1 if word.isalpha()))


for xyzzy in sent1:
    if xyzzy.endswith('l'):
        print(xyzzy)
        
for token in sent1:
    if token.islower():
        print(token, 'is a lowercase word')
    elif token.istitle():
        print(token, 'is a titlecase word')
    else:
        print(token, 'is punctuation')
        
tricky = sorted(w for w in set(text2) if 'cie' in w or 'cei' in w)
for word in tricky:
    print(word, end=' ')



















