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

from urllib2 import urlopen
url = "http://www.gutenberg.org/files/2554/2554.txt"
response = urlopen(url)
raw = response.read().decode('utf8')
type(raw)

tokens = word_tokenize(raw)
type(tokens)
len(tokens)
tokens[:10]

text = nltk.Text(tokens)
type(text)
text[1024:1062]
text.collocations()

raw.find("PART I")
raw.rfind("End of Project Gutenberg's Crime")
raw = raw[5338:1157746]
raw.find("PART I")

url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
html = urlopen(url).read().decode('utf8')
html[:60]

from bs4 import BeautifulSoup
raw = BeautifulSoup(html).get_text()
tokens = word_tokenize(raw)
tokens[:10]
tokens = tokens[110:390]
text = nltk.Text(tokens)
text.concordance('gene')

import feedparser
llog = feedparser.parse("http://languagelog.ldc.upenn.edu/nll/?feed=atom")
llog['feed']['title']
len(llog.entries)
post = llog.entries[4]
post.title
content = post.content[0].value
content[:100]
raw = BeautifulSoup(content).get_text()
word_tokenize(raw)

from nltk.corpus import gutenberg
raw = gutenberg.raw('melville-moby_dick.txt')
fdist = nltk.FreqDist(ch.lower() for ch in raw if ch.isalpha())
fdist.most_common(5)
[char for (char, count) in fdist.most_common()]

fdist.plot()

import re
wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]
[w for w in wordlist if re.search('ed$', w)]
# with ^..j..t..$, ^ marks start of word, j is 3rd and t is 6 and word length = 8
[w for w in wordlist if re.search('^..j..t..$', w)]
# j can be any letter and t 3 letters after, word length > 8
[w for w in wordlist if re.search('..j..t..', w)]

sum(1 for w in wordlist if re.search('e-?mail$', w))

[w for w in wordlist if re.search('^[g-o]+$', w)]

chat_words = sorted(set(w for w in nltk.corpus.nps_chat.words()))
[w for w in chat_words if re.search('^m+i+n+e+$', w)]
[w for w in chat_words if re.search('^[ha]+$', w)]

word = 'supercalifragilisticexpialidocious'
re.findall(r'[aeiou]', word)
len(re.findall(r'[aeiou]', word))

wsj = sorted(set(nltk.corpus.treebank.words()))
fd = nltk.FreqDist(vs for word in wsj
                   for vs in re.findall(r'[aeiou]{2,}', word))
fd.most_common(12)

regexp = r'^[AEIOUaeiou]+|[AEIOUaeiou]+$|[^AEIOUaeiou]'
def compress(word):
    pieces = re.findall(regexp, word)
    return ''.join(pieces)

english_udhr = nltk.corpus.udhr.words('English-Latin1')
print(nltk.tokenwrap(compress(w) for w in english_udhr[:75]))

rotokas_words = nltk.corpus.toolbox.words('rotokas.dic')
cvs = [cv for w in rotokas_words for cv in re.findall(r'[ptksvr][aeiou]', w)]
cfd = nltk.ConditionalFreqDist(cvs)
cfd.tabulate()

cv_word_pairs = [(cv, w) for w in rotokas_words
                 for cv in re.findall(r'[ptksvr][aeiou]', w)]
cv_index = nltk.Index(cv_word_pairs)
cv_index['su']
cv_index['po']

def stem(word):
    regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
    stem, suffix = re.findall(regexp, word)[0]
    return stem

raw = """DENNIS: Listen, strange women lying in ponds distributing swords 
is no basis for a system of government.  Supreme executive power derives from
a mandate from the masses, not from some farcical aquatic ceremony."""

tokens = word_tokenize(raw)
[stem(t) for t in tokens]

from nltk.corpus import gutenberg, nps_chat
moby = nltk.Text(gutenberg.words('melville-moby_dick.txt'))
moby.findall(r"<a>(<.*>)<man>")

chat = nltk.Text(nps_chat.words())
chat.findall(r"<.*><.*><bro>")
chat.findall(r"<l.*>{3,}")

from nltk.corpus import brown
hobbies_learned = nltk.Text(brown.words(categories=['hobbies', 'learned']))
hobbies_learned.findall(r"<\w*><and><other><\w*s>")















