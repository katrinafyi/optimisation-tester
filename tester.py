from typing import * 

from functools import lru_cache

import numpy as np
import scipy as sp 
import scipy.sparse as sps 
import scipy.sparse.linalg as spsl 


SecondOrderReturn = Tuple[float, np.array, np.array]

def least_squares(A: np.array, b: np.array, x: np.array):
    ATA = A.transpose() @ A
    return (
        np.sum(np.square(A @ x - b))/2, 
        ATA @ x - A.transpose() @ b,
        ATA
    )

def parabola(order: int, x: np.array):
    # assert order >= 2
    return (
        np.sum(np.power(x, order)), 
        np.multiply(order, np.power(x, order-1)), 
        np.diag(np.multiply(order*(order-1), np.power(x, order-2)).flatten())
    )

def exp_x_squared(x):
    return (
        -np.exp(-np.sum(np.power(x, 2))),
        2 * x * np.exp(-(np.power(x, 2))),
        None,
    )

def rosenbrock(x_tup):
    a = 1
    b = 100 
    x, y = x_tup
    return (
        (a-x)**2 + b*(y - x**2)**2,
        np.array((400*x**3 - 400*x*y + 2*x - 2, 200*(y-x**2))),
        np.array((
            (1200*x**2 - 400*y + 2, -400*x),
            (-400*x, 200)
        ))
    )

def gradient_descent(x, fun, data):
    alpha = 0.001
    fx, gx = fun(x)[:2]
    return x - alpha * gx, data


def nesterov_agd(x, fun, data):
    # https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/
    beta = 4

    # x_s = data.get('x_s', np.zeros_like(x))
    x_s = x
    y_s = data.get('y_s', np.zeros_like(x))
    lambda_s_minus_1 = data.get('lambda_s_minus_1', 0)

    lambda_s = (1 + np.sqrt(1 + 4 * lambda_s_minus_1**2)) / 2
    lambda_s_plus_1 = (1 + np.sqrt(1 + 4 * lambda_s**2)) / 2

    gamma_s = (1-lambda_s) / lambda_s_plus_1

    y_s_plus_1 = x_s - 1/beta * fun(x_s)[1]
    x_s_plus_1 = (1-gamma_s) * y_s_plus_1 + gamma_s * y_s
    
    data['y_s'] = y_s_plus_1
    data['lambda_s_minus_1'] = lambda_s

    return x_s_plus_1, data

def newton_matinv(x, fun, data):
    # print(Hx)
    _, gx, Hx = fun(x)[:3]
    return x - np.linalg.inv(Hx) @ gx, data

def newton_cg(x, fun, data):
    _, gx, H = fun(x)[:3]
    # solve Hp = -g
    p, status = spsl.cg(H, gx)
    return x - p, data

def test_optimiser(obj_fun, opt_fun, x0):
    x = x0
    data = {}
    print('starting at', x0.tolist())
    for i in range(200):
        prev_x = x
        x, data = opt_fun(x, obj_fun, data)
        fx, gx, Hx = obj_fun(x)

        print(str(i).ljust(3), fx, (gx).tolist(), (x).tolist())
        if np.any(np.isnan(x)):
            print('error: NaN')
            break
        if np.linalg.norm(gx) < 10e-6:
            print('converged???')
            break


if __name__ == "__main__":
    # obj_fun_lambda = lambda x: parabola(2, x+100000)
    obj_fun_lambda = rosenbrock
    start_pos = np.array((2, 1))
    for opt in (gradient_descent, nesterov_agd, newton_matinv, newton_cg):
        print('optimiser:', opt)
        test_optimiser(obj_fun_lambda, opt, start_pos)
        print()
        print()

    # spam_data = np.genfromtxt('spambase.csv', delimiter=',')
    # print(spam_data)