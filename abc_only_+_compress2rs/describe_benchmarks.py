import os


benchmark_dir = "./Benchmarks"
benchmark_name = "LGSynth"
output_file = open("LGSynth_description.csv","w")

for b in os.listdir(benchmark_dir):
	read_i = read_o = read_p = False
	bf = open(os.path.join(benchmark_dir, b),"r")
	while not(read_i and read_o):
		line = bf.readline()
		if '.i ' in line:
			num_inputs = line.split(" ")[1].strip('\n')
			read_i = True
		elif '.o ' in line:
			num_outputs = line.split(" ")[1].strip('\n')
			read_o = True

	bf.close()

	print(b, num_inputs, num_outputs, file = output_file)

output_file.close()
