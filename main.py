import numpy as np
import matplotlib.pylab as plt


def func2(x):
    return x[0] * x[0] + x[1] * x[1]


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        f_plus_del = f(x)
        x[idx] = tmp_val - h
        f_minus_del = f(x)
        grad[idx] = (f_plus_del - f_minus_del) / (2 * h)
        x[idx] = tmp_val

    return grad


print(numerical_gradient(func2, np.array([3.0, 4.0])))
print(numerical_gradient(func2, np.array([0.0, 2.0])))
print(numerical_gradient(func2, np.array([3.0, 0.0])))
