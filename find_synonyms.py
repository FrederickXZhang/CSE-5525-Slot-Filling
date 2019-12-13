#################################################################################
# Author: Frederick X. Zhang
# Usage: TBD
################################################################################

import sys
import numpy
import nltk

from nltk.corpus import wordnet as wn
	
	
class Synonym(object):
    def __init__(self, word):
        synonym=[]
        for syn in wn.synsets(word):
            for l in syn.lemmas():
                synonym.append(l.name())
                            
        print(set(synonym))
                            

        


if __name__ == "__main__":
    syno = Synonym(sys.argv[1])