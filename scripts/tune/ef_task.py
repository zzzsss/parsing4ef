from control import *
from os import system, popen
from sys import argv
import random

class efResult(Result):
    def __init__(self, l):
        super().__init__()
        self.l = l
        self.best_one = max(l)
    def __lt__(self, other):
        return self.best_one > other.best_one
    def __str__(self):
        return str(self.best_one) + " in " + str(self.l)

class efTask(OneTask):
    task_id = 0
    def execute(self, arg):
        efTask.task_id += 1
        # the dir
        conf_temp = "conf.txt"
        dir_name = "task%s" % efTask.task_id
        system("mkdir %s" % dir_name)
        # write conf
        s = ""
        with open(conf_temp) as fd:
            for l in fd:
                fs = l.split()
                if len(fs)>=1:
                    one = random.choice(fs)
                else:
                    one = ""
                s += one + "\n"
        with open("%s/_conf" % dir_name, 'w') as fd:
            fd.write(s)
        # execute (this one ignore arg)
        system("cd %s; bash ../../run.sh _conf >z.log 2>&1" % dir_name)
        # get results
        ret = popen("cat %s/z.log | grep '^zzzzz'" % dir_name).read()
        ret = ret.split()[1:]
        res = [float(i) for i in ret]
        return (arg, efResult(res))

if __name__ == "__main__":
    n = int(argv[1])
    x = TaskPool(efTask, workers=n)
    x._add_task([i for i in range(int(argv[2]))])    # max number of random select
    x.run()
