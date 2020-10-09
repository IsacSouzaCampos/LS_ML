from __future__ import print_function
import os
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
import platform

function_mappings = {
    'chi2': chi2,
    'f_classif': f_classif,
    'mutual_info_classif': mutual_info_classif,
}


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


def run_collapse_script(bencmark_dir, name_benchmark, collapse_file_path, outdir='.'):
    with open("%s/scriptCollapse" % outdir, "w") as f_script:
        print("read_pla %s/%s" % (bencmark_dir, name_benchmark), file=f_script)
        # print("strash", file=f_script)
        print("collapse", file=f_script)
        print("write_pla -m %s" % collapse_file_path, file=f_script)

    os.system("./abc -F %s/scriptCollapse" % outdir)


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


def run_optimized_espresso(full_path, final_name, outdir='Benchmarks_espresso'):
    if not os.path.exists(outdir):
        os.system(f'mkdir {outdir}')
    final_path = '%s/%s' % (outdir, final_name)
    print(f'espresso --fast {full_path} > {final_path}')
    os.system(f'espresso --fast {full_path} > {final_path}')
    return f'{final_path}'


def gen_aig(espresso_pla_path, outdir='AIG'):
    if not os.path.exists(outdir):
        os.system(f'mkdir {outdir}')
    final_path = '%s/%s' % (outdir, espresso_pla_path.split('/')[-1].replace('pla', 'aig'))
    with open('tmp/aig_maker_script', 'w') as aig_maker, open('compress2rs', 'r') as compress2rs:
        print(f'read_pla {espresso_pla_path}', file=aig_maker)
        print('strash', file=aig_maker)
        print(f'write_aiger {final_path}', file=aig_maker)

    os.system(f'./abc -F tmp/aig_maker_script')
    return f'{final_path}'


def compress2rs(aig):
    with open('tmp/compress2rs_script', 'w') as fout, open('compress2rs', 'r') as c2rs:
        print(f'read_aiger {aig}', file=fout)
        print(f'{c2rs.read()}', file=fout)
        print(f'write_aiger {aig}', file=fout)

    os.system('./abc -F tmp/compress2rs_script')


def run_aig(aig, pla, outdir="."):
    with open("%s/scriptRunAig" % outdir, "w") as fOut:
        print("&r %s" % aig, file=fOut)
        print("&ps", file=fOut)
        print("&mltest %s" % pla, file=fOut)
    os.system("./abc -F %s/scriptRunAig > %s/abc_output.txt" % (outdir, outdir))

    if platform.system() == "Darwin":
        os.system('sed -E "s/\x1B\[([0-9]{1,3}(;[0-9]{1,2})?)?[mGK]//g" %s/abc_output.txt > %s/abc_output_parsed.txt'
                  % (outdir, outdir))
    else:
        os.system('sed -r "s/\x1B\[([0-9]{1,3}(;[0-9]{1,2})?)?[mGK]//g" %s/abc_output.txt > %s/abc_output_parsed.txt'
                  % (outdir, outdir))

    with open("%s/abc_output_parsed.txt" % outdir, "r") as fIn:
        lines = fIn.readlines()
        # print(lines[26])
        ands = lines[4].split()[8]
        lev = lines[4].split()[11]
        # print(lines[29])
        acc_abc = float(lines[7].split()[9][1:])

    return ands, lev, acc_abc
