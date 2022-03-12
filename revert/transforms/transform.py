class Transform:

    def __init__(self, f): 
        self.eval = f

    def __matmul__(self, other): 
        return Transform(lambda t: self(other(t)))

    def __or__(self, other):
        return other @ self

    def __call__(self, x):
        return self.eval(x)
