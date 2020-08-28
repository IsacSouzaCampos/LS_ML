from __future__ import print_function
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree
from copy import deepcopy
import numpy as np
from math import sqrt, ceil
from itertools import combinations
from sklearn.model_selection import train_test_split, cross_val_score

import time

from itertools import combinations

import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from copy import deepcopy
from sklearn.tree import *
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from itertools import product

import platform

from os import listdir
from os.path import isfile, join
import numpy as np

function_mappings = {
    'chi2': chi2,
    'f_classif': f_classif,
    'mutual_info_classif': mutual_info_classif,
}


def run_collapse_script(bencmark_dir, name_benchmark, collapse_file_path, outdir='.'):
    with open("%s/scriptCollapse" % (outdir), "w") as f_script:
        print("read_pla %s/%s" % (bencmark_dir, name_benchmark), file=f_script)
        # print("strash", file=f_script)
        print("collapse", file=f_script)
        print("write_pla -m %s" % (collapse_file_path), file=f_script)

    os.system("./abc -F %s/scriptCollapse" % (outdir))


def parse_pla_header(folder, name_benchmark):
    with open("%s/%s" % (folder, name_benchmark), "r") as f_in:
        nr_inputs = nr_outputs = 0
        while True:
            line = f_in.readline().strip()
            if ".i " in line:
                nr_inputs = int(line.split()[1])
            if ".o " in line:
                nr_outputs = int(line.split()[1])

            if nr_inputs and nr_outputs:
                return nr_inputs, nr_outputs


def fillWithZeroes(X, X_new):
    X_new2 = deepcopy(X)
    i = 0
    for column in X.T:
        found = 0
        for compColumn in X_new.T:
            if np.array_equal(column, compColumn):
                found = 1
                break
        if found == 0:
            X_new2[:, i] = 0
        i += 1

    return X_new2


def pythonizeSOP(sop, classifier='other'):
    or_list = []
    expr = ''
    if sop == []:
        expr = '((x0) * !(x0))'
        return expr

    for ands in sop:
        and_list = []
        and_expr = '('
        for attr, negated in ands:
            if negated == 'true':
                and_list.append('not(x%s)' % (attr))
            else:
                and_list.append('(x%s)' % (attr))
        and_expr += ' and '.join(and_list)
        and_expr += ')'
        or_list.append(and_expr)
    expr = '(%s)' % (' or '.join(or_list))
    return expr


def pythonizeRF(sopRF):
    exprs = []
    for sopDT in sopRF:
        exprs.append(pythonizeSOP(sopDT))

    print(exprs)
    if exprs == []:
        finalExpr = '((x0) * !(x0))'
        return finalExpr

    nrInputsMaj = len(exprs)
    sizeTermMaj = int(ceil(nrInputsMaj / 2.0))
    ands = []

    for comb in combinations(exprs, sizeTermMaj):
        ands.append(" and ".join(comb))

    finalExpr = '(%s)' % (") or (".join(ands))

    return finalExpr


def trainMajorityRF(Xtr, ytr, Xte, yte, Xte2, yte2, num_trees=100, apply_SKB=0, apply_SP=0, score_f="chi2", thr=0.8,
                    k=10, percent=50, depth=10, useDefaultDepth=1):
    Xtr_new = deepcopy(Xtr)

    if apply_SKB == 1:
        selector = SelectKBest(function_mappings[score_f], k=k)
        selector.fit(Xtr, ytr)
        Xtr_new = selector.transform(Xtr)
        Xtr_new = fillWithZeroes(Xtr, Xtr_new)

    if apply_SP == 1:
        selector = SelectPercentile(function_mappings[score_f], percentile=percent)
        selector.fit(Xtr, ytr)
        Xtr_new = selector.transform(Xtr)
        Xtr_new = fillWithZeroes(Xtr, Xtr_new)

    num_feats_sub = int(sqrt(Xtr_new.shape[1]))
    num_feats = Xtr_new.shape[1]
    trees = []
    votes = []
    votes2 = []
    for i in range(num_trees):
        cols_idx = np.random.choice(range(num_feats), num_feats - num_feats_sub)

        Xtr_sub = np.array(Xtr_new)
        Xtr_sub[:, cols_idx] = 1
        tree = None
        if useDefaultDepth == 1:
            tree = DecisionTreeClassifier().fit(Xtr_sub, ytr)
        else:
            tree = DecisionTreeClassifier(max_depth=depth).fit(Xtr_sub, ytr)
        trees.append(tree)
        votes.append(tree.predict(Xte))
        votes2.append(tree.predict(Xte2))
    votes = np.array(votes).T
    votes2 = np.array(votes2).T
    final_vote = np.round(votes.sum(axis=1) / float(num_trees)).astype('int')
    final_vote2 = np.round(votes2.sum(axis=1) / float(num_trees)).astype('int')
    acc = (final_vote == yte).mean()
    acc2 = (final_vote2 == yte2).mean()

    return trees, acc, acc2


def eval_single(d, eqn_str):
    eqn_str_orig = eqn_str

    for i, d_ in enumerate(d):
        eqn_str = eqn_str.replace('x%d)' % (i), (str(d_) + ')'))

    return int(eval(eqn_str))


def eval_equation(eqn_str, data):
    X, y = data[:, :-1], data[:, -1]
    hits = 0
    ypred = np.apply_along_axis(eval_single, 1, X, eqn_str)
    return str((np.equal(ypred, y, dtype=int)).mean())


def treeToSOP(tree, featureNames):
    tree_ = tree.tree_
    featureName = [featureNames[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]

    ors = []

    if tree_.feature[0] == _tree.TREE_UNDEFINED:
        if np.argmax(tree_.value[0]) == 1:
            ors.append([['1', 'true']])
            ors.append([['1', 'false']])
        return ors

    def recurse(node, depth, expression):
        indent = "\t" * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = featureName[node]

            recurse(tree_.children_left[node], depth + 1, deepcopy(expression + [[name, 'true']]))

            recurse(tree_.children_right[node], depth + 1, deepcopy(expression + [[name, 'false']]))
        else:
            if np.argmax(tree_.value[node]) == 1:
                ors.append(deepcopy(expression))

    recurse(0, 1, [])

    return ors


def forestToSOP(forest, featureNames):
    sops = []
    for tree in forest:
        sop = treeToSOP(tree, featureNames)
        sops.append(sop)
    return sops


def trainTree(Xtr, ytr):
    tree = None

    tree = DecisionTreeClassifier().fit(Xtr, ytr)

    ypred_tree = tree.predict(Xtr)
    acc_tree = (ypred_tree == ytr).mean()

    return tree, acc_tree


def gen_eqn(nrInputs, example, expr, nameOut):
    os.system("mkdir -p EQNS")
    with open("EQNS/%s.eqn" % (nameOut), "w") as fOut:
        print("INORDER = %s;" % (" ".join(["x%d" % (a) for a in range(nrInputs)])), file=fOut)
        print("OUTORDER = z1;", file=fOut)
        print("%s" % (expr.replace("and", "*").replace("or", "+").replace("not", "!")))
        print("z1 = %s;" % (expr.replace("and", "*").replace("or", "+").replace("not", "!")), file=fOut)


def gen_eqn_aig(nrInputs, expr, baseName, outdir="."):
    os.system("mkdir -p EQNS")
    os.system("mkdir -p AIGS")
    os.system("mkdir -p OPT_AIGS")

    with open("EQNS/%s.eqn" % (baseName), "w") as fOut, open("%s/scriptGenAig" % (outdir), "w") as fOut2, open(
            "%s/scriptOptAig" % (outdir), "w") as fOut3:
        print("INORDER = %s;" % (" ".join(["x%d" % (a) for a in range(nrInputs)])), file=fOut)
        print("OUTORDER = z1;", file=fOut)
        print("z1 = %s;" % (expr.replace("and", "*").replace("or", "+").replace("not", "!")), file=fOut)
        print("read_eqn EQNS/%s.eqn" % (baseName), file=fOut2)
        print("strash", file=fOut2)
        print("write_aiger AIGS/%s.aig" % (baseName), file=fOut2)
        print("read_aiger AIGS/%s.aig" % (baseName), file=fOut3)
        print("refactor", file=fOut3)
        print("rewrite", file=fOut3)
        print("write_aiger OPT_AIGS/%s.aig" % (baseName), file=fOut3)

    os.system("./abc -F %s/scriptGenAig" % (outdir))
    os.system("./abc -F %s/scriptOptAig" % (outdir))
    return "OPT_AIGS/%s.aig" % (baseName)


def run_aig(aig, pla, outdir="."):
    with open("%s/scriptRunAig" % outdir, "w") as fOut:
        # print(f'read_aiger {aig}', file=fOut)
        # print('refactor', file=fOut)
        # print('rewrite', file=fOut)
        # print(f'{fin.read()[:-1]}', file=fOut)
        # print(f'write_aiger {aig}', file=fOut)
        print("&r %s" % aig, file=fOut)
        print("&ps", file=fOut)
        print("&mltest %s" % pla, file=fOut)
    os.system("./abc -F %s/scriptRunAig > %s/abc_output.txt" % (outdir, outdir))

    if platform.system() == "Darwin":
        os.system('sed -E "s/\x1B\[([0-9]{1,3}(;[0-9]{1,2})?)?[mGK]//g" %s/abc_output.txt > %s/abc_output_parsed.txt' % (outdir, outdir))
    else:
        os.system('sed -r "s/\x1B\[([0-9]{1,3}(;[0-9]{1,2})?)?[mGK]//g" %s/abc_output.txt > %s/abc_output_parsed.txt' % (outdir, outdir))

    with open("%s/abc_output_parsed.txt" % outdir, "r") as fIn:
        lines = fIn.readlines()
        # print(lines[26])
        ands = lines[4].split()[8]
        lev = lines[4].split()[11]
        # print(lines[29])
        acc_abc = float(lines[7].split()[9][1:])

    return ands, lev, acc_abc


# def plot_PDF(clf, feats = None, labels = None, numTrees = 3, classifier = 'forest'):
# 	if classifier == 'tree':
# 		dot_data = export_graphviz(clf, out_file=None, feature_names=feats, class_names = labels, filled=True, rounded=True, special_characters=True)
# 		graph = graphviz.Source(dot_data)
# 		graph.render("0")
# 	else:
# 		for i in range(numTrees):
# 			dot_data = export_graphviz(clf[i], out_file=None, feature_names=feats, class_names= labels, filled=True, rounded=True, special_characters=True)
# 			graph = graphviz.Source(dot_data)
# 			graph.render("%d" % (i))


def write_single_out_pla(pla, idx, outdir="tmp"):
    fout = open("%s/temp.pla" % outdir, "w")
    fin = open(pla, "r")
    total_lines = 0
    data_str = ''
    for line in fin.readlines():
        line = line.strip("\n")
        if '#' in line:
            continue
        elif '.e' in line:
            continue
        elif ".o " in line:
            print('.o 1\n.type fr', file=fout)
        elif ".ob" in line:
            continue
        elif ".ilb" in line:
            continue
        elif "." in line:
            print(line, file=fout)
        elif line[0] in ['0', '1']:
            lin = line.split(" ")
            data_str += lin[0] + ' ' + lin[1][idx] + '\n'
            total_lines += 1

    data_str = data_str.rstrip("\n")
    while total_lines < 64:
        data_str = data_str + '\n' + data_str
        total_lines *= 2

    print(data_str, file=fout)
    print('.e', file=fout)
    return '%s/temp.pla' % outdir


def run_espresso(full_path, final_name, outdir='Benchmarks_espresso'):
    if not os.path.exists(outdir):
        os.system(f'mkdir {outdir}')
    final_path = '%s/%s' % (outdir, final_name)
    print(f'espresso --fast {full_path} > {final_path}')
    os.system(f'espresso --fast {full_path} > {final_path}')
    return f'{final_path}'


def gen_pla_aig(espresso_pla_path, outdir='AIG'):
    if not os.path.exists(outdir):
        os.system(f'mkdir {outdir}')
    final_path = '%s/%s' % (outdir, espresso_pla_path.split('/')[-1].replace('pla', 'aig'))
    with open('aig_maker_script', 'w') as aig_maker:
        print(f'read_pla {espresso_pla_path}', file=aig_maker)
        print('strash', file=aig_maker)
        print(f'write_aiger {final_path}', file=aig_maker)

    os.system(f'./abc -F aig_maker_script')
    return f'{final_path}'
