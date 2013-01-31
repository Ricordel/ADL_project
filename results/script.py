#!/usr/bin/python

import sys

lines = open(sys.argv[1]).readlines()

for l in lines:
    function = l.split(":")[1].split(" ")[1][:-1]
    length, max_length = l.split(":")[2].split(",")[0], l.split(":")[3]
    length, max_length = int(length), int(max_length)

    percentage = float(length) / float(max_length)

    print("%s & %d & %d & %d & %f \\\\" % (function, length, max_length, max_length - length, percentage))
