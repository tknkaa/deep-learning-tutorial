import numpy as np
import matplotlib.pylab as plt


def func1(x):
    return 0.01 * x**2 + 0.1 * x


x = np.arange(0.0, 20.0, 0.1)
y = func1(x)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, y)
plt.savefig("out.png")


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


print(numerical_diff(func1, 5))
print(numerical_diff(func1, 10))
