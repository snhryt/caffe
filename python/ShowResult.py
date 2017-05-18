#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys, csv
import numpy as np

categories = np.loadtxt(sys.argv[1], str, delimiter=" ")
scores = np.load(sys.argv[2])
output_file = open(sys.argv[3], 'w')
csv_writer = csv.writer(output_file)

input_dir = str(sys.argv[2])
input_dir = input_dir[0: input_dir.rfind("/") + 1]
input_txtpath = input_dir + "TestFilenameList.txt"
with open(input_txtpath, 'r') as sub_input_file:
	filenames = sub_input_file.read().strip().split("\n")

top_k = 10
header = [""]
for i in range(0, top_k):
	header.append("rank" + str(i + 1))
	header.append("confidence[%]")
csv_writer.writerow(header)

for i in range(0, len(scores)):
	list_data = []
	print(filenames[i])
	list_data.append(filenames[i])
	prediction = zip(scores[i].tolist(), categories)
	prediction.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)
	for rank, (score, name) in enumerate(prediction[:top_k], start=1):
		if int(score * 100) == 0:
			break
		font = str(name)
		font = font[font.find(" '") + 2 : -1]
		font = font[0 : font.find("']")]
		print('#%d | %s | %4.1f%%' % (rank, font, score * 100))
		list_data.append(font)
		list_data.append("%4.1f" % (score * 100))
	csv_writer.writerow(list_data)
output_file.close()
