from control import *
from os import system, popen
from sys import argv
import random
from threading import Lock

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
    id_lock = Lock()
    def execute(self, arg):
        # counting
        efTask.id_lock.acquire()
        cur_id = efTask.task_id
        efTask.task_id += 1
        efTask.id_lock.release()
        # the dir
        dir_name = "task%s" % cur_id
        system("mkdir %s" % dir_name)
        # write conf
        s = self.get_conf(cur_id)
        if s is None:     # nothing if no confs
            return (arg, efResult([0.]))
        with open("%s/_conf" % dir_name, 'w') as fd:
            fd.write(s)
        # execute (this one ignore arg)
        system("cd %s; bash ../../run.sh _conf >z.log 2>&1" % dir_name)
        # get results
        ret = popen("cat %s/z.log | grep '^zzzzz'" % dir_name).read()
        ret = ret.split()[1:]
        res = [float(i) for i in ret]
        return (arg, efResult(res))

    def get_conf(self, cur_id):
        conf_temp = "conf.txt"
        s = ""
        with open(conf_temp) as fd:
            for l in fd:
                fs = l.split()
                if len(fs)>=1:
                    one = random.choice(fs)
                else:
                    one = ""
                s += one + "\n"
        return s

class efTaskSeq(efTask):
    def get_conf(self, cur_id):
        conf_temp = "conf.txt"
        s = ""
        num = cur_id
        with open(conf_temp) as fd:
            for l in fd:
                fs = l.split()
                length = len(fs)
                if len(fs)>=1:
                    one = fs[num%length]
                    num = num//length
                else:
                    one = ""
                s += one + "\n"
        return s

if __name__ == "__main__":
    n = int(argv[1])
    x = TaskPool({"rand":efTask, "seq":efTaskSeq}[argv[3]], workers=n)
    x._add_task([i for i in range(int(argv[2]))])    # max number of random select
    x.run()

# python3 ../ef/scripts/tune/ef_task.py 12 12 rand/seq
