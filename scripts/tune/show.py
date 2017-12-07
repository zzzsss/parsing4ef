#! /bin/python3

import os

def main():
    rank_file = "ranking.txt"
    for i, line in enumerate(open(rank_file)):
        fs = line.split()
        print("Read rank %s: %s" % (i+1, line.strip()))
        #os.system("awk 'NR==%s' %s" % (i+1, rank_file))
        os.system("cat task%s/_conf" % (int(fs[0])+1,))

main()
