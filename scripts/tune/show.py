#! /bin/python3

import os, sys

def main():
    dir = sys.argv[1]
    rank_file = dir+"/ranking.txt"
    for i, line in enumerate(open(rank_file)):
        fs = line.split()
        print("Read rank %s: %s" % (i+1, line.strip()))
        #os.system("awk 'NR==%s' %s" % (i+1, rank_file))
        os.system("cat %s/task%s/_conf" % (dir, int(fs[0])+1,))

main()
