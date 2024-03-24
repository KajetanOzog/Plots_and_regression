import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x ** 2

x = np.linspace(-5, 5, 200)
y = f(x)
plt.plot(x, y, '--k', ms=10)


def step_gradient_1d(x_current, learningRate):
    x_gradient = 2 * x_current
    new_x = x_current - learningRate * x_gradient

    plt.arrow(x_current, f(x_current), - (learningRate * x_gradient), -(f(x_current) - f(new_x)),
              head_width=0.05, head_length=0.5, ec="red")

    return new_x


def gradient_descent_runner_1d(starting_x, learning_rate, num_iterations):
    x = starting_x
    print(x)
    for i in range(num_iterations):
        x = step_gradient_1d(x, learning_rate)
        # print(x)
    return x


learning_rates = [0.001, 0.1, 0.2, 0.5, 0.9, 0.99, 0.999]
initial_x = 5
num_iterations = 300
for lr in learning_rates:
    x = gradient_descent_runner_1d(initial_x, lr, num_iterations)
    print(x)
    plt.title(f"Learning Rate {lr}")
    plt.show()
