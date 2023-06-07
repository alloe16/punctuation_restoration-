
from __future__ import division, print_function
from nltk.tokenize import word_tokenize

import nltk
import os
from io import open
import re
import sys

nltk.download("punkt")

NUM = "<NUM>"

EOS_PUNCTS = {".": ".PERIOD", "?": "?QUESTIONMARK", "!": "!EXCLAMATIONMARK"}
INS_PUNCTS = {",": ",COMMA", ";": ";SEMICOLON", ":": ":COLON", "-": "-DASH"}

forbidden_symbols = re.compile(r"[\[\]\(\)\/\\\>\<\=\+\_\*]")
numbers = re.compile(r"\d")
multiple_punct = re.compile(r"([\.\?\!\,\:\;\-])(?:[\.\?\!\,\:\;\-]){1,}")

is_number = lambda x: len(numbers.sub("", x)) / len(x) < 0.6

def process_line(line):

    tokens = word_tokenize(line)
    output_tokens = []

    for token in tokens:

        if token in INS_PUNCTS:
            output_tokens.append(INS_PUNCTS[token])
        elif token in EOS_PUNCTS:
            output_tokens.append(EOS_PUNCTS[token])
        elif is_number(token):
            output_tokens.append(NUM)
        else:
            output_tokens.append(token.lower())

    return " ".join(output_tokens) + " "
