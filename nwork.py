from numpy import *
h_glob = 10**(-4) # set global accuracy


class network(object):
    """
    Simple implementation of Neural Net for self-study purposes
    """
    def __init__(self, s):
        self.num_lay = len(s)
        self.s = s
        self.biases = [ random.randn(y, 1) for y in self.s[1:] ]
        self.w = [random.randn(y, x) for x, y in zip(s[:-1], s[1:])]


def sigmoid(x):
    return 1/ (1 + np.exp(-x))


def gradient(x, obj_func = sigmoid):
    """
    Function evaluating the n-dimensional Gradient vector by using the
    centered finite difference formula
    """
    f = obj_func
    h = h_glob
    dim = len(x)
    e = np.identity(dim)
    arr = np.zeros((1,dim))

    for i in range(dim):

        arr[0][i] = (f(x + h * e[:][i]) - f(x - h * e[:][i])) / (2*h)

    return arr


def hessian(x):
    """
    Function evaluating the Hessian of an n-dimensional
    objective function taking the Jacobian of the Gradient vector
    """

    h = h_glob
    if len(np.shape(x)) <= 1:
        dim = len(x)
    else:
        dim = len(x[0])
    e = np.identity(dim)
    arr = np.empty((dim, dim))
    
    for i in range(dim):
        arr[i][:] = np.array(((gradient(x + h * e[:][i]) - gradient(x - h * e[:][i])) / (2 * h)))
    return arr                
