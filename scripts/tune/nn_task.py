from control import *
from os import system, popen
from sys import argv

class nnResult(Result):
    def __init__(self, l):
        super().__init__()
        self.l = l
        self.best_one = max(l)
    def __lt__(self, other):
        return self.best_one > other.best_one
    def __str__(self):
        return str(self.best_one) + " in " + str(self.l)

class nnTask(OneTask):
    task_id = 0
    def execute(self, arg):
        nnTask.task_id += 1
        # the dir
        dir_name = "task%s" % nnTask.task_id
        system("mkdir task%s" % dir_name)
        # execute 
        system("cd %s; zt %s" % (dir_name, arg))
        # get results
        ret = popen("cat %s/z.log | grep '^zzzzz'" % log_file).read()
        ret = ret.split()[1:]
        res = [float(i) for i in ret]
        return (arg, nnResult(res))

if __name__ == "__main__":
    n = int(argv[1])
    x = TaskPool(nnTask, workers=n)
    x.run()
