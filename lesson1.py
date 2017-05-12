# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

f = open('airline-safety.csv', mode = 'rU')
file_string = f.read()
f.close()

file_string

# use a context manager to automatically close your file
with open('airline-safety.csv', mode = 'rU') as f:
    file_list = []
    for row in f:
        file_list.append(row)
    
file_list

# do the same thing using a list comprehension
with open('airline-safety.csv', mode ='rU') as f:
    file_list = [row for row in f]
    
file_list

split_str = 'hello DAT students'.split('e')
split_str

" ".join(split_str)
"e".join(split_str)

with open('airline-safety.csv', mode ='rU') as f:
    file_nested_list = [row.split(",") for row in f]

file_nested_list[1]

# do the same thing using the csv module
import csv
with open('/data/airline-safety.csv', mode ='rU') as f:
    file_nested_line = [row for row in csv.reader(f)]
    
file_nested_line[0:1]