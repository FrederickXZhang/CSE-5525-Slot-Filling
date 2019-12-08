#################################################################################
# Author: Frederick X. Zhang
# Usage: python3 unk_enhance.py SF-ID-Network-For-NLU/data/atis/train/seq.in.new SF-ID-Network-For-NLU/data/atis/train/seq.in.new.unk.outside SF-ID-Network-For-NLU/data/atis/train/seq.out 0.1 2 outside
################################################################################

import sys
import numpy
import nltk
import random

class UNKer(object):
    # priority can only be either entity, outside or full.
    def __init__(self, readfile, writefile, reffile, ratio=0.1, threshold=2, priority='full'):
        with open(writefile, 'w') as out:
            with open(reffile, 'r') as ref:
                with open(readfile, "r") as file:
                    r = ref.readlines()
                    f = file.readlines()
                    for i in range(len(r)):
                        temp_r = r[i].strip().split()
                        temp_f = f[i].strip().split()
                        
                        unked = temp_f.copy()
                        i=0
                        unk_num = 0
                        length = len(temp_r)
                        start_index = int(random.random()*length)
                        
                        for j in range(length):
                            token = temp_r[(start_index+j)%length]
                            if unk_num == threshold:
                                break
                            if token[0] != 'O':
                                if random.random() <= ratio and priority != 'outside':
                                    unked[(start_index+j)%length] = '_UNK' 
                                    unk_num += 1
                            elif token[0] == 'O':
                                if random.random() <= ratio and priority != 'entity':
                                    unked[(start_index+j)%length] = '_UNK' 
                                    unk_num += 1

                        out.write(" ".join(unked)+"\n")
                        


if __name__ == "__main__":
    unker = UNKer(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
