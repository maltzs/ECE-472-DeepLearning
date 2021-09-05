#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

N = 50
M = 10
batch_size = 16
num_iter = 750
learning_rate = 0.001
sigma_noise = 0.1


class Model(tf.Module):
    def __init__(self, num_gaussians):
        self.w = tf.Variable(tf.random.normal(shape=[num_gaussians, 1]))
        self.b = tf.Variable(tf.zeros(shape=[1, 1]))
        self.mu = tf.Variable(tf.linspace(0.0, 1.0, num_gaussians))
        self.sigma = tf.Variable(tf.ones(shape=[1, num_gaussians]))

    def __call__(self, x):
        gaussians = tf.exp(-(((x - self.mu) / self.sigma) ** 2))
        return tf.squeeze(gaussians @ self.w + self.b)


def main():
    index = np.arange(N)
    x_data = np.random.uniform(0.0, 1.0, size=(N, 1)).astype("float32")
    y_data = np.sin(2 * np.pi * x_data) + np.random.normal(
        scale=sigma_noise, size=(N, 1)
    )

    model = Model(M)
    optimizer = tf.optimizers.SGD(learning_rate=learning_rate)

    for i in range(num_iter):
        ind = np.random.choice(index, batch_size)
        x = x_data[ind]
        y = y_data[ind].flatten()
        with tf.GradientTape() as tape:
            y_hat = model(x)
            loss = 0.5 * (y - y_hat) ** 2

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    x_noiseless = np.linspace(0.0, 1.0, 100)
    y_noiseless = np.sin(2 * np.pi * x_noiseless)
    y_gaussians = np.exp(
        -(((x_noiseless.reshape(-1, 1) - model.mu.numpy()) / model.sigma.numpy()) ** 2)
    )
    y_regression = y_gaussians @ model.w.numpy() + model.b.numpy()

    plt.figure()
    plt.plot(
        x_data,
        y_data,
        "go",
        x_noiseless,
        y_noiseless,
        "b",
        x_noiseless,
        y_regression,
        "r--",
    )
    plt.xlim((0, 1))
    plt.ylim((-1.5, 1.5))
    plt.xlabel("x")
    y_label = plt.ylabel("y")
    y_label.set_rotation(0)
    plt.title("Sinewave regression")
    plt.savefig("sine.png", format="png")

    plt.figure()
    plt.plot(x_noiseless, y_gaussians)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel("x")
    y_label = plt.ylabel("y")
    y_label.set_rotation(0)
    plt.title("Gaussian basis functions")
    plt.savefig("gaussians.png", format="png")


if __name__ == "__main__":
    main()
