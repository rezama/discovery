from multiprocessing import Pool

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
    
    return a

if __name__ == '__main__':
    print f((A(2), 2))
    objs = []
    for i in range(10):
        objs.append((A(i), i))
    pool = Pool(processes=4)              # start 4 worker processes
#    result = pool.apply_async(f, [10])     # evaluate "f(10)" asynchronously
#    print result.get(timeout=1)           # prints "100" unless your computer is *very* slow
    print pool.map(f, objs)          # prints "[0, 1, 4,..., 81]"
    for i in range(10):
        print objs[i][0].value
    