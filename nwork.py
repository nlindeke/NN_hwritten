from numpy import *

class network(object):
    """
    Simple implementation of Neural Net for self-study purposes
    """
    def __init__(self, s):
        self.num_lay = len(s)
        self.s = s
        self.biases = [ random.randn(y, 1) for y in self.s[1:] ]
        self.w = [random.randn(y, x) for x, y in zip(s[:-1], s[1:])]
        
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [ training_data[k:k + mini_batch_size] for k in xrange(0, n, mini_batch_size) ]    

            for mini_batch in mini_batches:
                
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)        