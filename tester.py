from typing import * 

import numpy as np

def least_squares(A: np.ndarray, b: np.array, x: np.array):
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

def test_fun_with_method(obj_fun, opt_fun):
    pass

if __name__ == "__main__":
    for x in (parabola(2, [2, 1])):
        print(x)