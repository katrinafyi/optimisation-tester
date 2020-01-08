from typing import * 

from functools import lru_cache

import numpy as np
import scipy as sp 
import scipy.sparse as sps 
import scipy.sparse.linalg as spsl 

# from sympy import lambdify
import matplotlib.pyplot as plt


SecondOrderReturn = Tuple[float, np.array, np.array]

def least_squares(A: np.array, b: np.array, x: np.array):
    return (
        np.sum(np.square(A @ x - b))/2, 
        A.transpose() @ (A@x) - A.transpose() @ b,
        A.transpose() @ A
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
    # x,y = x_tup[:,0], x_tup[:,1]
    x,y = x_tup

    return (
        (a-x)**2 + b*(y - x**2)**2,
        np.array((400*x**3 - 400*x*y + 2*x - 2, 200*(y-x**2))),
        np.array((
            (1200*x**2 - 400*y + 2, -400*x),
            (-400*x, 200)
        ))
    )

def gradient_descent(x, fun, data):
    alpha = 0.000001
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


def newton_nesterov(x, fun, data):
    # https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/

    # x_s = data.get('x_s', np.zeros_like(x))
    x_s = x
    y_s = data.get('y_s', np.zeros_like(x))
    lambda_s_minus_1 = data.get('lambda_s_minus_1', 0)

    lambda_s = (1 + np.sqrt(1 + 4 * lambda_s_minus_1**2)) / 2
    lambda_s_plus_1 = (1 + np.sqrt(1 + 4 * lambda_s**2)) / 2

    gamma_s = (1-lambda_s) / lambda_s_plus_1

    _, gx, H = fun(x_s)[:3]
    # print(x_s.shape)
    # print(H.shape)
    # print(gx.shape)
    y_s_plus_1 = x_s - spsl.cg(H, gx)[0]
    x_s_plus_1 = (1-gamma_s) * y_s_plus_1 + gamma_s * y_s
    
    data['y_s'] = y_s_plus_1
    data['lambda_s_minus_1'] = lambda_s

    return x_s_plus_1, data

def test_optimiser(obj_fun, opt_fun, x0):
    x = x0
    data = {}
    print('starting at', x0.tolist())
    history = [x]
    for i in range(200):
        prev_x = x
        x, data = opt_fun(x, obj_fun, data)
        history.append(x)
        fx, gx, Hx = obj_fun(x)


        print(str(i).ljust(3), fx, (gx).tolist(), (x).tolist())
        if np.any(np.isnan(x)):
            print('error: NaN')
            break
        if np.linalg.norm(gx) < 10e-6:
            print('converged???')
            break
    return history


if __name__ == "__main__":
    # obj_fun_lambda = lambda x: parabola(2, x+100000)
    # obj_fun_lambda = lambda x: parabola(2, x)
    obj_fun_lambda = rosenbrock

    # obj_fun_lambda = lambda x: least_squares(
    #     np.array(((1, 2, 3), (10, 31, 21), (12, 41, 21))),
    #     np.array((10, 20, 30)),
    #     x)
    start_pos = np.array([200, 10])
    # (gradient_descent, nesterov_agd, newton_matinv, newton_cg)
    for opt in (gradient_descent, newton_matinv, newton_cg, newton_nesterov):
        print('optimiser:', opt)
        test_optimiser(obj_fun_lambda, opt, start_pos)

        # a = np.linspace(-50, 50, 1000)
        # b = np.linspace(-30, 30, 1000)
        # ex, ey = np.meshgrid(a, b)
        # exy = np.array(list(zip(ex, ey)))
        # # print(ex, ey)
        # # ef = lambdify((x, y), f.func(), "numpy")
        # c = plt.contour(ex, ey, obj_fun_lambda(exy, 1)[0])

        # history = test_optimiser(obj_fun_lambda, opt, start_pos)
        # plt.scatter([i[0] for i in history], [i[1] for i in history])
        # plt.plot([i[0] for i in history], [i[1] for i in history])

        # plt.show()

        print()
        print()

    # spam_data = np.genfromtxt('spambase.csv', delimiter=',')
    # print(spam_data)