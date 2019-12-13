#################################################################################
# Author: Frederick X. Zhang
# Usage: python3 stemming.py SF-ID-Network-For-NLU/data/atis/train/seq.in SF-ID-Network-For-NLU/data/atis/train/seq.in.new
################################################################################

import sys
import numpy
import nltk

from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

class Stemmer(object):
    def __init__(self, readfile, writefile):
        with open(writefile, 'w') as out:
            with open(readfile, "r") as file:
                for line in file:
                    tokens = line.strip().split()
                    tags = pos_tag(tokens)
                    update_tags = tags.copy()
                    i=0
                    
                    for tagged in tags:
                        lemma = ''
                        lemma_help = self.get_wordnet_pos(tagged[1])
                        if lemma_help == '':
                            lemma = lemmatizer.lemmatize(tagged[0])
                        else:
                            lemma = lemmatizer.lemmatize(tagged[0], pos=lemma_help)
                        update_tags[i] = lemma
                        i += 1
                    out.write(" ".join(update_tags)+"\n")
        

    def get_wordnet_pos(self, tag):
    
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return ''


if __name__ == "__main__":
    stem = Stemmer(sys.argv[1], sys.argv[2])