#################################################################################
# Author: Frederick X. Zhang
# Usage: TBD
################################################################################

import sys
import numpy
import nltk

from nltk.corpus import wordnet as wn
	
	
class Synonym(object):
    def __init__(self, readfile, writefile):
        with open(writefile, 'w') as out:
            with open(readfile, "r") as file:
                for line in file:
                    synonyms = {}
                    tokens = line.strip().split()
                    
                    for token in tokens:
                        temp = []
                        for syn in wordnet.synsets(token):
                            for l in syn.lemmas():
                                temp.append(l.name())
                                
                        synonyms[token] = set(temp)
                        
                        #To be discussed
                        #There will be infinitely many combinations (Combinatorial explosion)
                        #We should have a smart way to filter the top k
                    
                    out.write()
        


if __name__ == "__main__":
    syno = Synonym(sys.argv[1], sys.argv[2])