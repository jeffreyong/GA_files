#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 12:06:22 2017

@author: Work
"""

import nltk
nltk.corpus.gutenberg.fileids()

emma = nltk.corpus.gutenberg.words('austen-emma.txt')
len(emma)

emma = nltk.Text(nltk.corpus.gutenberg.words('austen-emma.txt'))
emma.concordance("surprise")

from nltk.corpus import gutenberg
gutenberg.fileids()
emma = gutenberg.words('austen-emma.txt')

for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
    print(round(num_chars/num_words), round(num_words/num_sents), round(num_words/num_vocab), fileid)

macbeth_sentences = gutenberg.sents('shakespeare-macbeth.txt')
macbeth_sentences
macbeth_sentences[1116]
longest_len = max(len(s) for s in macbeth_sentences)
[s for s in macbeth_sentences if len(s) == longest_len]

from nltk.corpus import webtext
for fileid in webtext.fileids():
    print(fileid, webtext.raw(fileid)[:65], '...')
    
from nltk.corpus import nps_chat
chatroom = nps_chat.posts('10-19-20s_706posts.xml')
chatroom[123]

from nltk.corpus import brown
brown.categories()

brown.words(categories='news')
brown.words(fileids=['cg22'])
brown.sents(categories=['news', 'editorial', 'reviews'])

news_text = brown.words(categories='news')
fdist = nltk.FreqDist(w.lower() for w in news_text)
modals = ['can', 'could', 'may', 'might', 'must', 'will']
for m in modals:
    print(m + ':', fdist[m], end=' ')
    
fiction_text = brown.words(categories='fiction')
fdist = nltk.FreqDist(w.lower() for w in fiction_text)
wh_words = ['who', 'what', 'where', 'when', 'where', 'why']
for w in wh_words:
    print(w + ":", fdist[w], end=' ')
    
cfd = nltk.ConditionalFreqDist(
        (genre, word)
        for genre in brown.categories()
        for word in brown.words(categories=genre))
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will', 'who', 'what', 'where', 'when']
cfd.tabulate(conditions=genres, samples=modals)

from nltk.corpus import reuters
reuters.fileids()
reuters.categories()

reuters.categories('training/9865')
reuters.categories(['training/9865', 'training/9880'])
reuters.fileids(['barley', 'corn'])

reuters.words('training/9865')[:14]
reuters.words(['training/9865', 'training/9880'])
reuters.words(categories=['barley', 'corn'])

from nltk.corpus import inaugural
inaugural.fileids()
[fileid[:4] for fileid in inaugural.fileids()]

cfd = nltk.ConditionalFreqDist(
        (target, fileid[:4])
        for fileid in inaugural.fileids()
        for w in inaugural.words(fileid)
        for target in ['america', 'citizen']
        if w.lower().startswith(target))

cfd.plot()

nltk.corpus.cess_esp.words()
nltk.corpus.floresta.words()
nltk.corpus.indian.words('hindi.pos')
nltk.corpus.udhr.fileids()
nltk.corpus.udhr.words('Javanese-Latin1')[11:]

from nltk.corpus import udhr
languages = ['Chickasaw', 'English', 'German_Deutsch', 'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist(
        (lang, len(word))
        for lang in languages
        for word in udhr.words(lang + '-Latin1'))
cfd.plot(cumulative=True)

raw = gutenberg.raw("burgess-busterbrown.txt")
raw[1:20]




































