import numpy as np 
import sympy as sy 


"""
A class that symbolically computes gradients and Hessians of 3D multivariable
functions, and returns them as lambdas as well as evaluates them at a given
point specified by vals.
"""
class Function:
    def __init__(self, f, vars):
        # Compute function, gradient and Hessian
        self.f = f
        self.vars = vars
        self.g = sy.Matrix([f.diff(i) for i in vars])
        self.h = sy.Matrix([[g.diff(i) for i in vars] for g in self.g])

    """
    Return function as a function.
    """
    def func(self):
        return self.f

    """
    Return gradient as a sympy matrix.
    """
    def gradient(self):
        return self.g

    """
    Return Hessian as a matrix function.
    """
    def hessian(self):
        return self.h

    """
    Returns the function evaluation, updating values if necessary.
    """
    def eval_func(self, vals):
        vals_dict = dict(zip(self.vars, vals))
        return self.f.subs(vals_dict).evalf()
    
    """
    Returns the gradient evaluation, updating values if necessary.
    """
    def eval_grad(self, vals):
        vals_dict = dict(zip(self.vars, vals))
        return np.array(self.g.subs(vals_dict)).astype(float)

    """
    Returns the Hessian evaluation, updating values if necessary.
    """
    def eval_hess(self, vals):
        vals_dict = dict(zip(self.vars, vals))
        return np.array(self.h.subs(vals_dict)).astype(float)

if __name__ == '__main__':
    x, y = sy.symbols('x y')
    f = Function(x**2 + y**2, (x, y))
    print(f.f)
    print(f.g)
    print(f.h)
    print(f.eval_grad((0, 1)))