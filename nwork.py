from numpy import *

class network(object):
    def __init__(self, s):
        self.num_lay = len(s)
        self.s = s
        self.biases = [ random.randn(y,1) for y in self.s[1:] ]
        self.w = [random.randn(y, x) for x, y in zip(s[:-1], s[1:])]
        