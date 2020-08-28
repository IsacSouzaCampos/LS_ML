from __future__ import print_function
from sys import argv
from utilities import *

os.system('python3 clean.py')

benchmark_dir = "Benchmarks"
collapse_dir = "Benchmarks_collapse"

exec_collapse = 1
exec_training = 1
MAX_WIDTH = 16  # maximum input width (storage/computing restrictions)
if len(argv) > 1:
    MAX_WIDTH = int(argv[1])

# PART 1 --- COLLAPSING INPUT FILES TO OBTAIN COMPLETE TRUTH TABLES #############
time_table = []
benchmark_total_time = []
os.system("mkdir -p tmp")

if exec_collapse:
    name_benchmarks = [f for f in listdir(benchmark_dir) if isfile(join(benchmark_dir, f))]
    os.system("mkdir -p %s" % collapse_dir)

    name_new_benchmarks = []

    for name_benchmark in name_benchmarks:
        nr_inputs, nr_outputs = parse_pla_header(benchmark_dir, name_benchmark)

        if nr_inputs > MAX_WIDTH:
            continue

        name_split = name_benchmark.split(".")[0]

        time_i = time.time()
        collapse_file_path = "%s/%s_collapse_%di_%do.pla" % (collapse_dir, name_split, nr_inputs, nr_outputs)
        run_collapse_script(benchmark_dir, name_benchmark, collapse_file_path, outdir="tmp")
        time_f = time.time()
        time_table.append([name_benchmark, "collapse", nr_inputs, str(time_f - time_i).replace('.', ':')])

# PART 2 --- ESPRESSO REDUCTION AND AIG TRANSLATION #############
seed = 0
np.random.seed(seed)

train_file_dir = collapse_dir
train_file_list = os.listdir(train_file_dir)

f_out = open("training_results.csv", "w")
total_time_output = open('benchmarks_total_time.csv', 'w')
print(",".join(["benchmark", "nb inputs", "nb outputs", "output idx", "accuracy ABC", "nb ands", "aig depth"]),
      file=f_out)

for pla_path in train_file_list:

    if ".pla" not in pla_path:
        continue

    path_split = pla_path.split(".")[0].split("_")
    nr_inputs, nr_outputs = path_split[-2], path_split[-1]

    base_name = '_'.join(path_split[:-3])
    full_path = os.path.join("%s/%s" % (train_file_dir, pla_path))
    nr_inputs, nr_outputs = int(nr_inputs.strip('i')), int(nr_outputs.strip('o'))

    X, Y = np.loadtxt(full_path, dtype='str', comments=".", skiprows=1, unpack=True)
    X = [list(x) for x in X]
    X = np.asarray(X).astype('uint8')
    Y = [list(y) for y in Y]
    Y = np.asarray(Y).astype('uint8')

    feature_names = list(map(str, list(range(X.shape[1]))))

    num_feats = len(feature_names)

    total_time = 0
    for idx in range(Y.shape[1]):
        tmp_name = base_name + "_out%d" % idx

        y = Y[:, idx]

        time_i = time.time()
        pla_single = write_single_out_pla(full_path, idx)
        time_f = time.time()
        total_time += time_f - time_i
        time_table.append([pla_path, "write_single_out_pla", num_feats, str(time_f - time_i).replace('.', ':')])

        time_i = time.time()
        espresso = run_espresso(pla_single, full_path.split('/')[-1])
        time_f = time.time()
        total_time += time_f - time_i
        time_table.append([pla_path, "run_espresso", num_feats, str(time_f - time_i).replace('.', ':')])

        time_i = time.time()
        aig = gen_pla_aig(espresso)
        time_f = time.time()
        total_time += time_f - time_i
        time_table.append([pla_path, "gen_pla_aig", num_feats, str(time_f - time_i).replace('.', ':')])

        time_i = time.time()
        ands, levs, acc_abc = run_aig(aig, pla_single, outdir="tmp")
        time_f = time.time()
        total_time += time_f - time_i
        time_table.append([pla_path, "run_aig", num_feats, str(time_f - time_i).replace('.', ':')])

        time_table.append([pla_path, 'total_time', num_feats, str(total_time).replace('.', ':')])

        print(",".join([str(x) for x in [pla_path, nr_inputs, nr_outputs, idx, acc_abc, ands, levs]]), file=f_out)
    print(",".join([str(x) for x in [pla_path, str(total_time).replace('.', ':')]]), file=total_time_output)

np.savetxt("time_table.csv", time_table, fmt="%s", delimiter=",", header="benchmark,function,nb inputs,time/call (s)")

f_out.close()