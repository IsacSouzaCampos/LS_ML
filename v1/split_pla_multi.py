from __future__ import print_function
import os
from os import listdir
from os.path import isfile, join

folder = "Benchmarks"
folderCollapse = "Benchmarks_collapse"
folderSplits = "Benchmarks_collapse_split"

execCollapse = 1

if execCollapse == 1:
	nameBenchmarks = [f for f in listdir(folder) if isfile(join(folder, f))]
	os.system("mkdir -p %s" % (folderCollapse))

	nameNewBenchmarks = []

	for nameBenchmark in nameBenchmarks[:2]:
		nrInputs = 0
		with open("%s/%s" % (folder, nameBenchmark), "r") as fIn:
			while True:
				line = fIn.readline().strip()
				if ".i" in line:
					nrInputs = int(line.split()[1])
					break

		nameSplit = nameBenchmark.split(".")[0]

		with open("scriptCollapse", "w") as fScript:
			print("read_pla %s/%s" % (folder, nameBenchmark), file=fScript)
			#print("strash", file=fScript)
			print("collapse", file=fScript)
			print("write_pla -m %s/%s_collapse.pla" % (folderCollapse, nameSplit), file=fScript)

		os.system("./abc -F scriptCollapse")

nameNewBenchmarks = [f for f in listdir(folderCollapse) if isfile(join(folderCollapse, f)) and '.' != f[0]]

for nameBenchmark in nameNewBenchmarks:
	nameSplit = nameBenchmark.split(".")[0]

	os.system("mkdir -p %s/%s" % (folderSplits, nameSplit))
	nrInputs = 0
	nrOutputs = 0
	inputs = []
	outputs = []
	nrLines = 0

	with open("%s/%s" % (folderCollapse, nameBenchmark), "r") as fIn:
		lines = fIn.readlines()
		idxBeginData = 0
		for line in lines:
			idxBeginData += 1
			if ".i " in line:
				nrInputs = int(line.strip().split()[1])
			if ".o " in line:
				nrOutputs = int(line.strip().split()[1])
			if ".ilb" in line:
				inputs = line.strip().split()[1:]
			if ".ob" in line:
				outputs = line.strip().split()[1:]
			if ".p" in line:
				nrLines = int(line.strip().split()[1])
				break

		for _out in range(nrOutputs):
			with open("%s/%s/%s_%d.pla" % (folderSplits, nameSplit, nameSplit, _out), "w") as fOut:
				print(".i %d" % (nrInputs), file=fOut)
				print(".o 1", file=fOut)
				print(".ilb %s" % (" ".join(inputs)), file=fOut)
				print(".ob %s" % (outputs[_out]), file=fOut)
				if nrLines < 64:
					print(".p 64", file=fOut)
				else:
					print(".p %d" % (nrLines), file=fOut)
				for line in lines[idxBeginData:-1]:
					splitLine = line.strip().split()
					#print splitLine
					print("%s %s" % (splitLine[0], splitLine[1][_out]), file=fOut)
					if nrLines < 64:
						for i in range(int(64.0/nrLines)-1):
							print("%s %s" % (splitLine[0], splitLine[1][_out]), file=fOut)
				print(".e", file=fOut)
