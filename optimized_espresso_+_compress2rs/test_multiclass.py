from __future__ import print_function
from sys import argv
from utilities import *

benchmark_dir = "Benchmarks"
collapse_dir = "Benchmarks_collapse"

exec_collapse = 1
exec_training = 1
MAX_WIDTH = 16 # maximum input width (storage/computing restrictions)
if len(argv) > 1:
	MAX_WIDTH = int(argv[1])

## PART 1 --- COLLAPSING INPUT FILES TO OBTAIN COMPLETE TRUTH TABLES #############
time_table = []
os.system("mkdir -p tmp")

if exec_collapse:
	name_benchmarks = [f for f in listdir(benchmark_dir) if isfile(join(benchmark_dir, f))]
	os.system("mkdir -p %s" % (collapse_dir))

	name_new_benchmarks = []

	for name_benchmark in name_benchmarks:
		nr_inputs, nr_outputs = parse_pla_header(benchmark_dir, name_benchmark)

		if nr_inputs > MAX_WIDTH:
			continue

		name_split = name_benchmark.split(".")[0]
		
		time_i = time.time()
		collapse_file_path = "%s/%s_collapse_%di_%do.pla" % (collapse_dir, name_split, nr_inputs, nr_outputs)
		run_collapse_script(benchmark_dir, name_benchmark, collapse_file_path, outdir = "tmp")
		time_f = time.time()
		time_table.append(["collapse", nr_inputs, time_f - time_i])



## PART 2 --- DEC. TREE TRAINING AND AIG TRANSLATION #############

seed = 0
np.random.seed(seed)

train_file_dir= collapse_dir
train_file_list = os.listdir(train_file_dir)

f_out = open("training_results.csv", "w")
print(",".join(["benchmark", "nb inputs", "nb outputs", "output idx", "accuracy tree", "accuracy ABC", "nb ands", "aig depth", "sop"]), file=f_out)

for pla_path in train_file_list:

	if ".pla" not in pla_path: continue

	path_split = pla_path.split(".")[0].split("_")
	nr_inputs, nr_outputs = path_split[-2], path_split[-1]

	base_name = '_'.join(path_split[:-3])
	full_path = os.path.join("%s/%s" % (train_file_dir, pla_path))
	nr_inputs, nr_outputs = int(nr_inputs.strip('i')), int(nr_outputs.strip('o'))

	X, Y = np.loadtxt(full_path, dtype='str', comments=".", skiprows = 1, unpack = True)
	X = [list(x) for x in X]
	X = np.asarray(X).astype('uint8')
	Y = [list(y) for y in Y]
	Y = np.asarray(Y).astype('uint8')
	
	feature_names = list(map(str, list(range(X.shape[1]))))

	num_feats = len(feature_names)
	
	for idx in range(Y.shape[1]):
		tmp_name = base_name + "_out%d" % (idx)

		y = Y[:,idx]
		time_i = time.time()
		tree, acc_tree = trainTree(X, y)
		time_f = time.time()
		time_table.append(["trainTree", num_feats, time_f - time_i])
		
		time_i = time.time()
		sop_tree = treeToSOP(tree, feature_names)
		time_f = time.time()
		time_table.append(["treeToSOP", num_feats, time_f - time_i])
		
		time_i = time.time()
		expr_tree = pythonizeSOP(sop_tree, classifier = 'tree')
		time_f = time.time()
		time_table.append(["pythonizeSOP", num_feats, time_f - time_i])

		time_i = time.time()
		aig = gen_eqn_aig(nr_inputs, expr_tree, tmp_name, outdir = "tmp")
		time_f = time.time()
		time_table.append(["gen_eqn_aig", num_feats, time_f - time_i])

		time_i = time.time()
		pla_single = write_single_out_pla(full_path, idx)
		time_f = time.time()
		time_table.append(["write_single_out_pla", num_feats, time_f - time_i])

		time_i = time.time()
		ands, levs, acc_abc = run_aig(aig, pla_single, outdir = "tmp")
		time_f = time.time()
		time_table.append(["run_aig", num_feats, time_f - time_i])

		print(",".join([str(x) for x in [pla_path, nr_inputs, nr_outputs, idx, acc_tree, acc_abc, ands, levs, expr_tree]]), file=f_out)

np.savetxt("time_table.csv", time_table, fmt = "%s", delimiter = ",", header="function,nb inputs,time/call (s)")

f_out.close()
