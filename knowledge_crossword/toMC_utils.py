import pandas as pd
import numpy as np
import copy
import re


def camel_split(text):
    # find all capital letters
    # and insert an blank before each one, then convert
    # the whole string to lower case
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', text).lower()

def replace_underscore(text): # 2
    return text.replace("_", " ")

def remove_wordnet(text): # 1
    return re.sub(r'^wordnet_(\w+)_\d+$', r'\1', text)

def ABC_123(b, blanks): # A - blank 1
    if b in blanks:
        return "blank " + str(ord(b.upper()) - ord('A') + 1)
    else:
        return b

def num_ABC(i): # 0 - A, 1 - B, 2 - C
    return chr(i + ord('A'))


def hrt_to_constraints(head, relation, tail):
    x = "Constraints: "
    for h, r, t in zip(head, relation, tail):
        # h = replace_underscore(remove_wordnet(ABC_123(h, blanks)))
        # t = replace_underscore(remove_wordnet(ABC_123(t, blanks)))
        h = replace_underscore(remove_wordnet(h))
        t = replace_underscore(remove_wordnet(t))
        x += "("+h+", "+camel_split(r)+", "+t+"); "
    x = x[:-2] + ".\n"
    return x

def hrt_to_K(K_tri):
    hints = "Knowledge: "
    for v in K_tri.values():
        for h,r,t in v:
            h = replace_underscore(remove_wordnet(h))
            t = replace_underscore(remove_wordnet(t))
            hints += "("+h+", "+camel_split(r)+", "+t+"); "
    hints = hints[:-2]+".\n"
    return hints

def opt_to_options(opt, blanks):
    x = "Options:\n"
    for b in blanks:
            x += b + ": "
            for i in range(len(opt[b])):
                ABC = chr(i + ord('A'))
                x += ABC + ". " + replace_underscore(remove_wordnet(opt[b][i])) + ", "
            x = x[:-2] + "\n"
        
    return x



def hrt_to_MC_zeroshot(h,r,t, blanks, options):
    x = "Instruction: Pick the correct answer for each blank that satisfies all the given constraints.\n"
    x += "Desired format: blank i: Z ...\n"
    x += hrt_to_constraints(h,r,t)
    cur_options = opt_to_options(options, blanks)
    x += cur_options
    x += "Answer:"
    return x

