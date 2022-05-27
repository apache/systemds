import numpy as np
from itertools import zip_longest

M = 15
N = 1000
a = np.array(np.round(np.random.uniform(-50, 50, (N,M)), 0), dtype=np.int32)

def sign(a):
    return " + " if a >= 0 else " - "

def make_string(v):
    x = sign(v[-1]) + str(np.abs(v[-1]))
    for i in range(len(v)-1):
        x = x +  sign(v[i]) + str(np.abs(v[i]))
    return x


with open("data.csv", "w+") as f:
    for i in range(N):
        c = np.random.randint(2,M)
        s = make_string(a[i,:c])
        v = np.sum(a[i,:c]) % 51
        line = str(i) + ";" + s + ";" + str(v) + "\n"
        print(line)
        f.write(line)
