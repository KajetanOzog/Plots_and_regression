import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
plt.show()
X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print(theta_best)
X_new = np.linspace(0, 2, num=1000).reshape(-1, 1)
y_predict = X_new * theta_best[0] + theta_best[1]
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
reg = LinearRegression()
reg.fit(X, y)
y_predict_new = reg.predict(X_new)
plt.plot(X_new, y_predict_new, "g-")
plt.axis((0, 2, 0, 15))
plt.show()


#222222222222222

#BGD
def step_gradient(theta, learningRate, X, Y, m):
    gradient = 2/m * X.T.dot(X.dot(theta) - Y)
    new_theta = theta - learningRate * gradient
    return new_theta


def gradient_descent_runner(starting_theta, learning_rate, num_iterations, X, Y, m):
    x = starting_theta
    theta_path = []
    for i in range(num_iterations):
        x = step_gradient(x, learning_rate, X, Y, m)
        theta_path.append(x.ravel())
    return x, theta_path


eta = 0.1
n_iterations = 1000
m = 100
theta_0 = np.random.randn(2,1)
X_b = np.c_[np.ones((100, 1)), X]
theta, theta_path_bgd = gradient_descent_runner(theta_0, eta, n_iterations, X_b, y, m)
theta_path_bgd = np.array(theta_path_bgd)



#SGD


def stochastic_gradient_descent_runner(starting_theta, learning_rate, num_iterations, X, Y, m):
    theta = starting_theta
    theta_path = []
    for _ in range(num_iterations):
        for i in range(len(X)):
            prediction = np.dot(X[i], theta)
            error = prediction - y[i]
            gradient = (error * X[i]) / len(X)
            theta = (theta.T - learning_rate * gradient.T).T
        theta_path.append(theta.ravel())
    return theta, theta_path


np.random.seed(42)
eta = 0.1  # learning rate
n_iterations = 1000
m = 100
X_b = np.c_[np.ones((100, 1)), X]

theta, theta_path_sgd = stochastic_gradient_descent_runner(theta_0, eta, n_iterations, X_b, y, m)
theta_path_sgd = np.array(theta_path_sgd)


#MGD

def mini_batch_gradient_descent_runner(starting_theta, learning_rate, num_iterations, batch_size, X, Y, m):
    theta = starting_theta
    theta_path = []
    for _ in range(num_iterations):
        for batch_start in range(0, m, batch_size):
            batch_end = min(batch_start + batch_size, m)
            batch_indices = np.random.choice(range(m), size=batch_size, replace=False)
            X_batch = X[batch_indices]
            Y_batch = Y[batch_indices]
            gradient = (2 / batch_size) * np.dot(X_batch.T, np.dot(X_batch, theta) - Y_batch)
            theta = theta - learning_rate * gradient
        theta_path.append(theta.ravel())

    return theta, theta_path


n_iterations = 1000
minibatch_size = 20
m = 100
X_b = np.c_[np.ones((100, 1)), X]

theta, theta_path_mgd = mini_batch_gradient_descent_runner(theta_0, eta, n_iterations, minibatch_size, X_b, y, m)
theta_path_mgd = np.array(theta_path_mgd)

plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], "b", label="BGD")
plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], "r", label="MGD")
plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], "g", label="SGD")
plt.legend()
plt.show()