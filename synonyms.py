#################################################################################
# Author: Frederick X. Zhang
# Usage: TBD
################################################################################

import sys
import numpy
import nltk

from nltk.corpus import wordnet as wn
	
	
class Synonym(object):
    def __init__(self, readfile):
        with open("test.out", 'w') as out:
            with open(readfile, "r") as file:
                f = file.readlines()
                length = len(f)
                i=0
                
                while (i<length):
                    sentence = f[i].strip().split()
                    entity = f[i+1].strip().split()
                    intent = f[i+2].strip().split()
                    
                    inner_len = len(sentence)
                    sent_temp=[]
                    entity_temp=[]
                    synonym=[]
                    
                    for l in range(inner_len):
                        token = sentence[l]
                        synonym=[]
                        if token[0]=='_':
                            
                            for syn in wn.synsets(token[1:]):
                                for l in syn.lemmas():
                                    synonym.append(l.name())
                            
                            synonym = set(synonym)
                            
                            temp = token.split('_')
                            temp_length = len(temp)-1
                            sent_temp.extend(temp[1:])
                            entity_temp.append('B-'+str(entity[l]))
                            for t in range(1:temp_length):
                                entity_temp.append('I-'+str(entity[l]))
                            
                            
                                    
                            synonyms[token] = set(temp)
                        
                    
                    # out.write()
        


if __name__ == "__main__":
    syno = Synonym(sys.argv[1])