import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm


fun = lambda x, y: x ** 2 - y ** 2
derivative_x = lambda x: 2 * x
derivative_y = lambda y: 2 * y


fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1, projection='3d')
ax2 = fig.add_subplot(2, 1, 2)


X = np.arange(-7, 7, 0.25)
Y = np.arange(-7, 7, 0.25)
X, Y = np.meshgrid(X, Y)
Z = fun(X, Y)
surf = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0.01, antialiased=True, alpha=0.3)
cs = ax2.contour(X, Y, Z)
ax2.plot([5], [5], "o")
ax2.set_xlim([-7, 7])
ax2.set_ylim([-7, 7])


def step_gradient(x_current, y_current, learning_rate):
    x_gradient = derivative_x(x_current)
    y_gradient = derivative_y(y_current)
    new_x = x_gradient - learning_rate * x_gradient
    new_y = y_gradient - learning_rate * y_gradient

    ax2.arrow(x_current, y_current, - (learning_rate * x_gradient), - (learning_rate * y_gradient), head_width=0.05,
              head_length=0.5, ec="red")
    ax1.quiver(x_current, y_current, (fun(x_current, y_current)), - (learning_rate * x_gradient),
               - (learning_rate * y_gradient), (-(fun(x_current, y_current) - fun(new_x, new_y))))

    return new_x, new_y


def gradient_descent(x_starting, y_starting, learning_rate, iterations):
    x = x_starting
    y = y_starting
    for i in range(iterations):
        x, y = step_gradient(x, y, learning_rate)
    return x, y


learning_rate = 0.9
initial_x = 0
initial_y = 5
num_iterations = 5
[x, y] = gradient_descent(initial_x, initial_y, learning_rate, num_iterations)
ax1.plot([initial_x], [initial_y], [fun(initial_x, initial_y)], "ok")
ax2.axis("equal")
plt.show()


