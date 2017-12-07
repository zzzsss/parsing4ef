from threading import Thread, Lock
import random, sys, time

class OneTask:
    def execute(self, arg):
        raise("No implementation")

class Result:
    def __lt__(self, other):
        raise("No implementation")
    def __str__(self):
        raise("No implementation")

class Worker:
    def __init__(self):
        self.is_running = False
        self.result = (None, None)      # task-arg => Result
    def _start(self, task, arg):
        self.is_running = True
        self.result = task().execute(arg)     # task
        self.is_running = False
    def start(self, task, arg):
        Thread(target=self._start, args=(task, arg)).start()
    def fetch(self):
        tmp = self.result
        self.result = (None, None)
        return tmp

class InputReader:
    def __init__(self):
        self.lines = []
        self.lock = Lock()
        self.flag = True
        self.t = Thread(target=self._input)
        self.t.start()
    def _input(self):
        while self.flag:
            try:
                one = input()
                self.lock.acquire()
                self.lines.append(one)
                self.lock.release()
            except EOFError:
                print("EOF")
                return
    def getlines(self):
        self.lock.acquire()
        l = self.lines
        self.lines.clear()
        self.lock.release()
        return l
    def end(self):
        self.flag = False

# the running tasks center
class TaskPool:
    _iden_type = str
    def __init__(self, task, workers=1):
        self.task = task
        self.task_results = {}          # iden => task (None means is running)
        self.task_all = set()           # set of iden
        self.threads = workers
        self.workers_pool = []
        self.reader = InputReader()
        self.result_file = "ranking.txt"
        self.config_file = "config.txt"
    def _add_task(self, l):
        if isinstance(l, list):
            for x in l:
                self.task_all.add(x)
        elif isinstance(l, TaskPool._iden_type):
            self.task_all.add(l)
        else:
            raise("Illegal"+l)
    def _add_file(self, fn):
        try:
            with open(fn) as f:
                for l in f:
                    self._add_task(l)
        except FileNotFoundError as e:
            print("ERROR: No config-file " + fn)
    def _loop(self):
        flag_new_finish = False
        # 1. read cmd from input
        lines = self.reader.getlines()
        for l in lines:
            fileds = l.split(":")
            try:
                if fileds[0] == "workers":
                    self.threads = int(fileds[1])
                else:
                    raise("unknown")
            except Exception as e:
                print("CMD-wrong "+e.args+" of "+l)
        # 2. re-arrange workers
        new_pool = []
        for w in self.workers_pool:
            if w.is_running:
                new_pool.append(w)
            else:   # get result
                r = w.fetch()
                if r[0] is not None:
                    flag_new_finish = True
                    print("- Finish task " + str(r[0]) + " with " + str(r[1]))
                    self.task_results[r[0]] = r[1]
        self.workers_pool = new_pool
        # 3. assign new ones
        for t in self.task_all:
            if t not in self.task_results:
                # find a new worker
                if len(self.workers_pool) < self.threads:
                    print("- Start task " + str(t))
                    self.task_results[t] = None
                    self.workers_pool.append(Worker())
                    self.workers_pool[-1].start(self.task, t)
                else:
                    break
        # 4. write result file
        if flag_new_finish:
            rlist = []
            for a, b in self.task_results.items():
                if b is not None:
                    rlist.append((a, b))
            if len(rlist) > 0:
                rlist.sort(key=lambda x: x[1])
                with open(self.result_file, "w") as f:
                    for a, b in rlist:
                        f.write(str(a)+" "+str(b)+"\n")
        return len(self.workers_pool) != 0
    def run(self):
        self._add_file(self.config_file)
        while self._loop():
            time.sleep(1.0)
        self.reader.end()
        print("OK")

# simple ones
class SimpleResult(Result):
    def __init__(self, n):
        self.n = n
    def __lt__(self, other):
        return self.n < other.n
    def __str__(self):
        return str(self.n)

class SimpleTask:
    def execute(self, arg):
        time.sleep(int(arg))
        return arg, SimpleResult(random.random())

if __name__ == '__main__':
    # for test
    x = TaskPool(SimpleTask, workers=2)
    x._add_task([1,2,3,4])
    x.run()
