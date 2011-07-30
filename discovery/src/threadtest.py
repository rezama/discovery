from multiprocessing import Pool
import time

class A:

    def __init__(self, value=1, create_child=True):
        self.value = value
        if create_child:
            self.child = A(value, create_child=False)
        
    def __repr__(self):
        return str(self.value) + '-' + str(self.child.value)

def f((a, i)):
    a.value += 1
    a.child.value += 11
    if a.value == 5:
        time.sleep(1)
    
    return a

if __name__ == '__main__':
    print f((A(2), 2))
    objs = []
    obj_values = {}
    for i in range(10):
        obj = (A(i), i)
        obj_values[obj] = i
        objs.append((A(i), i))
    pool = Pool(processes=4)              # start 4 worker processes
#    result = pool.apply_async(f, [10])     # evaluate "f(10)" asynchronously
#    print result.get(timeout=1)           # prints "100" unless your computer is *very* slow
    result = pool.map(f, objs)          # prints "[0, 1, 4,..., 81]"
    print result
#    print obj_values[result[0]]
    for i in range(10):
        print objs[i][0].value
    