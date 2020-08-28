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

function_mappings = {
	'chi2': chi2,
	'f_classif': f_classif,
	'mutual_info_classif': mutual_info_classif,
}

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
			X_new2[:,i] = 0
		i += 1

	return X_new2
				

def pythonizeSOP(sop, classifier = 'other'):
	or_list = []
	expr = ''
	if sop == []:
		expr = '((x0) * !(x0))'
		return expr

	for ands in sop:
		and_list = []
		and_expr = '('
		for attr,negated in ands:
			if negated == 'true':
				and_list.append('not(x%s)' %  (attr))
			else:
				and_list.append('(x%s)' %  (attr))
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
	sizeTermMaj = int(ceil(nrInputsMaj/2.0))
	ands = []

	for comb in combinations(exprs, sizeTermMaj):
		ands.append(" and ".join(comb))

	finalExpr = '(%s)' % (") or (".join(ands))

	return finalExpr

def trainMajorityRF(train_data, test_data, test2_data, num_trees = 100, apply_SKB = 0, apply_SP = 0, score_f = "chi2", thr = 0.8, k = 10, percent = 50, depth = 10, useDefaultDepth= 1):
	Xtr, ytr = train_data[:,:-1], train_data[:,-1]
	Xte, yte = test_data[:,:-1], test_data[:,-1]
	Xte2, yte2 = test2_data[:,:-1], test2_data[:,-1]
	Xtr_new = deepcopy(Xtr)

	if apply_SKB == 1:
		selector = SelectKBest(function_mappings[score_f], k = k)
		selector.fit(Xtr, ytr)
		Xtr_new = selector.transform(Xtr)
		Xtr_new = fillWithZeroes(Xtr, Xtr_new)

	if apply_SP == 1:
		selector = SelectPercentile(function_mappings[score_f], percentile = percent)
		selector.fit(Xtr, ytr)
		Xtr_new = selector.transform(Xtr)
		Xtr_new = fillWithZeroes(Xtr, Xtr_new)

	num_feats_sub = int(sqrt(Xtr_new.shape[1]))
	num_feats = Xtr_new.shape[1]
	trees = []
	votes = []
	votes2 = []
	for i in range(num_trees):
		cols_idx = np.random.choice(range(num_feats),num_feats - num_feats_sub)

		Xtr_sub = np.array(Xtr_new)
		Xtr_sub[:,cols_idx] = 1
		tree = None
		if useDefaultDepth == 1:
			tree = DecisionTreeClassifier().fit(Xtr_sub, ytr)
		else:
			tree = DecisionTreeClassifier(max_depth = depth).fit(Xtr_sub, ytr)
		trees.append(tree)
		votes.append(tree.predict(Xte))
		votes2.append(tree.predict(Xte2))
	votes = np.array(votes).T
	votes2 = np.array(votes2).T
	final_vote = np.round(votes.sum(axis=1)/float(num_trees)).astype('int')
	final_vote2 = np.round(votes2.sum(axis=1)/float(num_trees)).astype('int')
	acc = (final_vote == yte).mean()
	acc2 = (final_vote2 == yte2).mean()

	return trees, acc, acc2

def eval_single( d, eqn_str):
	eqn_str_orig = eqn_str

	for i, d_ in enumerate(d):
		eqn_str = eqn_str.replace('x%d)' % (i), (str(d_)+')'))

	return int(eval(eqn_str))

def eval_equation(eqn_str, data):
	X, y = data[:,:-1], data[:,-1]
	hits = 0
	ypred =  np.apply_along_axis(eval_single, 1,  X, eqn_str)
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

def trainTree(train_data):

	Xtr, ytr = train_data[:,:-1], train_data[:,-1]

	tree = None

	tree = DecisionTreeClassifier().fit(Xtr, ytr)

	ypred_tree = tree.predict(Xtr)
	acc_tree = (ypred_tree == ytr).mean()

	return tree, acc_tree

def gen_eqn(folderC50, example, expr, nameOut):
	os.system("mkdir -p EQNS")
	with open("EQNS/%s.eqn" % (nameOut), "w") as fOut:
		numLines = sum(1 for line in open("%s/%s.names" % (folderC50, example), "r"))
		nrInputs = numLines - 4
		print("INORDER = %s;" % (" ".join(["x%d" % (a) for a in range(nrInputs)])), file=fOut)
		print("OUTORDER = z1;", file=fOut)
		print("%s" % (expr.replace("and", "*").replace("or", "+").replace("not", "!")))
		print("z1 = %s;" % (expr.replace("and", "*").replace("or", "+").replace("not", "!")), file=fOut)

def gen_eqn_aig(folderC50, expr, baseName):
	os.system("mkdir -p EQNS")
	os.system("mkdir -p AIGS")
	os.system("mkdir -p OPT_AIGS")

	with open("EQNS/%s.eqn" % (baseName), "w") as fOut, open("scriptGenAig", "w") as fOut2, open("scriptOptAig", "w") as fOut3:
		numLines = sum(1 for line in open("%s/%s.names" % (folderC50, baseName), "r"))
		nrInputs = numLines - 3
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

	os.system("./abc -F scriptGenAig")
	os.system("./abc -F scriptOptAig")

def run_aig(folderPla, baseName):
	with open("scriptRunAig", "w") as fOut:
		print("&r OPT_AIGS/%s.aig" % (baseName), file=fOut)
		print("&ps", file=fOut)
		print("&mltest %s/%s.pla" % (folderPla, baseName), file=fOut)
	os.system("./abc -F scriptRunAig > teste.txt")
	if platform.system() == "Darwin":
		os.system('sed -E "s/\x1B\[([0-9]{1,3}(;[0-9]{1,2})?)?[mGK]//g" teste.txt > teste2.txt')
	else:
		os.system('sed -r "s/\x1B\[([0-9]{1,3}(;[0-9]{1,2})?)?[mGK]//g" teste.txt > teste2.txt')
	with open("teste2.txt", "r") as fIn:
		lines = fIn.readlines()
		ands = lines[4].split()[8]
		print(lines)
		acc_abc = float(lines[7].split()[8])/float(lines[7].split()[2])

	return ands, acc_abc

def plot_PDF(clf, feats = None, labels = None, numTrees = 3, classifier = 'forest'):
	if classifier == 'tree':
		dot_data = export_graphviz(clf, out_file=None, feature_names=feats, class_names = labels, filled=True, rounded=True, special_characters=True)
		graph = graphviz.Source(dot_data)
		graph.render("0")
	else:
		for i in range(numTrees):
			dot_data = export_graphviz(clf[i], out_file=None, feature_names=feats, class_names= labels, filled=True, rounded=True, special_characters=True)
			graph = graphviz.Source(dot_data)
			graph.render("%d" % (i))


seed = 0

folderC50 = "c50_files"

listToTest = os.listdir(folderC50)

print(listToTest)
fOut = open("results.csv", "w")
print(",".join(["Benchmark", "Accuracy_tree", "Accuracy_ABC", "Ands", "Equation"]), file=fOut)
time_table = []

for path in listToTest:

	#if "02-adder_col_collapse" in path or "bw_collapse" in path or "squar5_collapse" in path or "rd53_collapse" in path: continue

	if ".data" not in path: continue

	base_name = path.split(".")[0]
	c50f_data = base_name + '.data'
	folderPla = "Benchmarks_collapse_split/%s" % ("_".join(base_name.split("_")[:-1]))

	print(folderPla)
	print(base_name)

	train_data = np.loadtxt("%s/%s" % (folderC50, c50f_data), dtype='int', delimiter=',')
	featureNames = list(map(str, list(range(train_data.shape[1]))))

	if len(featureNames) == 6 or len(featureNames) == 4: continue

	nameCfg = base_name
	np.random.seed(seed)
	num_feats = len(featureNames) -1
	
	time_i = time.time()
	tree, acc_tree = trainTree(train_data)
	time_f = time.time()
	time_table.append(["trainTree", num_feats, time_f - time_i])
	
	time_i = time.time()
	sop_tree = treeToSOP(tree, featureNames)
	time_f = time.time()
	time_table.append(["treeToSOP", num_feats, time_f - time_i])
	
	time_i = time.time()
	expr_tree = pythonizeSOP(sop_tree, classifier = 'tree')
	time_f = time.time()
	time_table.append(["pythonizeSOP", num_feats, time_f - time_i])

	time_i = time.time()
	gen_eqn_aig(folderC50, expr_tree, base_name)
	time_f = time.time()
	time_table.append(["gen_eqn_aig", num_feats, time_f - time_i])
	
	time_i = time.time()
	ands, acc_abc = run_aig(folderPla, base_name)
	time_f = time.time()
	time_table.append(["run_aig", num_feats, time_f - time_i])

	print(",".join([str(x) for x in [path, acc_tree, acc_abc, ands, expr_tree]]), file=fOut)

np.savetxt("time_table.csv", time_table, fmt = "%s")

fOut.close()
