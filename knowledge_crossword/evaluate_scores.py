import odqa_utils as utils
import pandas as pd
import numpy as np
import networkx as nx
import numpy as np
import re
import json
import argparse
import csv
import logging
from tqdm import tqdm
import os
import glob
import time
import string

from odqa_utils import single_ans_em, single_ans_f1

def write_data(filename, data):
    with open(filename, 'a') as fout:
        for sample in data:
            fout.write(json.dumps(sample))
            fout.write('\n')

def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def response_split(response, blanks, args):
    if args.mode in ["LtM", "CoT"]:
        so = response.find("Therefore, ")
        if so == -1:
            return None
        response = response[so:]

    ind_dic = dict()
    response_ans = {}
    for b in blanks:
        bb = b+":"
        index = response.find(bb)
        if index == -1:
            return None
        ind_dic[b] = index
    sorted_dict = dict(sorted(ind_dic.items(), key=lambda item: item[1]))

    l = list(sorted_dict.items())
    for i in range(len(l)):
        b, ind = l[i]
        start = ind+2
        if i < len(l)-1:
            end = l[i+1][1]
        else:
            end = len(response)
        response_ans[b] = response[start:end]




    # for i in range(0,len(ind)-1):
    #     if ind[i] >= ind[i+1]:
    #         print("answer: Not ascending order!")
    #         print(response)
    #         return None
    # ind.append(len(response)+1)
    # for i in range(len(blanks)):
    #     b = blanks[i]
    #     start = ind[i]+3
    #     end = ind[i+1]
    #     a = response[start:end]
    #     a = a.rstrip()
    #     a = a.rstrip(',')
    #     response_ans[b] = a
    return response_ans



def full_credit(response, blanks, ans):
    # ans: [(a1,b1),(a2,b2)]
    response_ans = response_split(response, blanks, args)
    if response_ans == None:
        return 0
    cur_high = 0
    for a in ans:
        cur_score_list = [single_ans_f1(response_ans[blanks[i]], a[i]) for i in range(len(blanks))]
        cur_score = sum(cur_score_list)/len(cur_score_list)
        cur_high = max(cur_high, cur_score)
    return cur_high

def partial_credit(response, blanks, ans_separate, deg_separate, ans, total_degree):
    # ans_separate: {"A":[],"B":[]}
    response_ans = response_split(response, blanks, args)
    if response_ans == None:
        return 0
    cur_high = 0
    full_cre = full_credit(response, blanks, ans)
    part_cre = 0
    part_degree = 0
    for i in range(len(blanks)):
        if blanks[i] not in ans_separate:
            continue
        cur_part_score = max([single_ans_f1(response_ans[blanks[i]], ans_separate[blanks[i]][j]) for j in range(len(ans_separate[blanks[i]]))])
        part_cre += cur_part_score * deg_separate[blanks[i]]
        part_degree += deg_separate[blanks[i]]
    res = (part_cre + total_degree * full_cre) / (part_degree + total_degree)
    return res


def main(args):
    preds = read_data(args.pred)
    golds = read_data(args.gold)
    ids = []
    partial = []
    full = []
    binary = []
    threshold_6 = []
    print(len(preds), len(golds))
    for i in tqdm(range(len(preds))):    
        p = preds[i]
        g = golds[i]
        assert p['id'] == g['id']
        ids.append(p["id"])
        response = p['response']
        blanks = g['blanks']
        ans_separate = g['ans_res']
        deg_separate = g['deg_res']
        ans = g['ans_all']
        total_deg = g['num_edges']
        cur_partial = partial_credit(response, blanks, ans_separate, deg_separate, ans, total_deg)
        cur_full = full_credit(response, blanks, ans)
        cur_binary = int(cur_full)
        cur_thre = int(cur_full >= 0.6)
        partial.append(cur_partial)
        full.append(cur_full)
        binary.append(cur_binary)
        threshold_6.append(cur_thre)

    
    filename = args.mode+"_score.txt"
    out_file = os.path.join(args.out_dir, filename)
    scores = {'id':ids, 'partial':partial, 'full':full, 'binary':binary, 'threshold_6':threshold_6}
    df = pd.DataFrame(scores)
    df.to_csv(out_file, sep="\t", header=True, index = False)

    filename = args.mode+"_summary.txt"
    filename = os.path.join(args.out_dir, filename)
    avg_partial = sum(partial)/len(ids)
    avg_full = sum(full)/len(ids)
    avg_binary = sum(binary)/len(ids)
    avg_thre = sum(threshold_6)/len(ids)
    print("Writing summary...")
    with open(filename, 'w') as file:
        # Write the content to the file
        file.write("partial: {}\n".format(avg_partial))
        file.write("full: {}\n".format(avg_full))
        file.write("binary: {}\n".format(avg_binary))
        file.write("threshold_6: {}\n".format(avg_thre))


def mean_score(out_dir):
    df = pd.read_csv(out_dir,sep='\t',header=None,names =["id","partial", "full", "binary", "threshold_6"])

    partial = df["partial"].tolist()
    # print(partial)
    full = df["full"].tolist()
    binary = df["binary"].tolist()
    threshold_6 = df["threshold_6"].tolist()
    a,b,c,d = 0,0,0,0
    non_zero = 0
    for i in range(1, len(partial)):
        if float(partial[i]) == float(0):
            continue
        else:
            non_zero += 1
            a += float(partial[i])
            b +=float(full[i])
            c +=float(binary[i])
            d +=float(threshold_6[i])
    print(a/non_zero,b/non_zero,b/non_zero,d/non_zero)
    print(non_zero)

    # print(partial, full, binary, threshold_6)

    # partial = df.iloc[:,1].mean()
    # full = df.iloc[:,2].mean()
    # binary = df.iloc[:,3].mean()
    # threshold_6 = df.iloc[:,4].mean()
    # print(partial, full, binary, threshold_6)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default=None, type=str, required=True, help="dataset:path to KG file or the folder containing all KG files;irrelevant_node_filter:path to KC_dataset.jsonl")
    parser.add_argument("--gold", default=None, type=str, required=True, help="path to gold answers")
    parser.add_argument("--mode", default="", type=str, required=False, choices=['0shot', '0shot_K', '5shot','5shot_K','LtM','LtM_K','CoT','CoT_K'])
    parser.add_argument("--out_dir", default=None, type=str, required=True)
    args = parser.parse_args()
    main(args)
    # mean_score("/home/data/wwangbw/YAGO/KC_prompt_result/p_5shot_nohint/5shot_nohint_score.txt")
    # mean_score("/home/data/wwangbw/YAGO/KC_prompt_result/p_5shot_hint/5shot_hint_score.txt")
    # mean_score("/home/data/wwangbw/YAGO/KC_prompt_result/p_LtM/LtM_score.txt")








