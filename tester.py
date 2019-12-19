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
    # assert order >= 2
    return (
        np.sum(np.power(x, order)), 
        np.multiply(order, np.power(x, order-1)), 
        np.diag(np.multiply(order*(order-1), np.power(x, order-2)).flatten())
    )

def gradient_descent(x, fun, data):
    alpha = 0.1
    fx, gx = fun(x)[:2]
    return x - alpha * gx, data

def nesterov_agd(x, fun, data):
    alpha = 0.1 
    beta = 0.9

    w_t = x
    m_t_minus_1 = data.get('m_t_minus_1', np.zeros_like(x))

    w_star = w_t - alpha * m_t_minus_1
    g_star = fun(w_star)[1]

    m_t = beta * m_t_minus_1 + (1-beta) * g_star 

    w_t_plus_1 = w_t - alpha * m_t

    data['m_t_minus_1'] = m_t

    return w_t_plus_1, data

def newton_matinv(x, fun, data):
    # print(Hx)
    _, gx, Hx = fun(x)[:3]
    return x - np.linalg.inv(Hx) @ gx, data

def newton_cg(x, fx, gx, Hx, data):
    pass

def test_optimiser(obj_fun, opt_fun, x0):
    x = x0
    data = {}
    for i in range(100):
        prev_x = x
        x, data = opt_fun(x, obj_fun, data)
        
        fx, gx, Hx = obj_fun(x)
        print(i, fx, (gx).tolist(), (x).tolist())
        if np.any(np.isnan(x)):
            print('error: NaN')
            break
        if np.allclose(x, prev_x, atol=0.0000000001):
            print('converged???')
            break


if __name__ == "__main__":
    test_optimiser(lambda x: parabola(2, x+1000), nesterov_agd, np.zeros((2,1)))
    print()
    print()
    print()
    test_optimiser(lambda x: parabola(2, x+1000), gradient_descent, np.zeros((2,1)))