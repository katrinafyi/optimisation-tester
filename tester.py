from typing import * 

import numpy as np

SecondOrderReturn = Tuple[float, np.array, np.array]

def least_squares(A: np.array, b: np.array, x: np.array):
    ATA = A.transpose() @ A
    return (
        np.sum(np.square(A @ x - b))/2, 
        ATA @ x - A.transpose() @ b,
        ATA
    )

def parabola(order: int, x: np.array):
    assert order >= 2
    return (
        np.sum(np.power(x, order)), 
        np.multiply(order, np.power(x, order-1)), 
        np.diag(np.multiply(order*(order-1), np.power(x, order-2)))
    )

def gradient_descent(obj_fun: Callable[[np.array], ], x):
    alpha = 0.1
    fx, gx = obj_fun(x)[:2]
    return x - alpha * gx, fx, gx

def test_optimiser(obj_fun, opt_fun, x0):
    x = x0
    for i in range(100):
        x, fx, gx = opt_fun(obj_fun, x)
        print(i, fx)
        print(gx)
        print(x)


if __name__ == "__main__":
    test_optimiser(lambda x: parabola(2, x+10), gradient_descent, np.zeros((2,1)))