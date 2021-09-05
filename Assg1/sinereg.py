#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from absl import flags, app

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_samples", 50, "Number of samples in dataset")
flags.DEFINE_integer("batch_size", 16, "Number of samples per iteration")
flags.DEFINE_integer("num_iter", 750, "Number of SGD iterations")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for SGD")
flags.DEFINE_integer("random_seed", 12345, "Random seed")
flags.DEFINE_float("sigma_noise", 0.1, "Random noise std")
flags.DEFINE_integer("num_gaussians", 10, "Number of gaussian basis functions")


class Data:
    def __init__(self, num_samp, sigma, random_seed):
        np.random.seed(random_seed)

        self.index = np.arange(num_samp)
        self.x = np.random.uniform(0.0, 1.0, size=(num_samp, 1)).astype("float32")
        self.y = np.sin(2 * np.pi * self.x) + np.random.normal(
            scale=sigma, size=(num_samp, 1)
        )

    def get_batch(self, batch_size):
        ind = np.random.choice(self.index, batch_size)
        return self.x[ind], self.y[ind].flatten()


class Model(tf.Module):
    def __init__(self, num_gaussians):
        self.w = tf.Variable(tf.random.normal(shape=[num_gaussians, 1]))
        self.b = tf.Variable(tf.zeros(shape=[1, 1]))
        self.mu = tf.Variable(tf.linspace(0.0, 1.0, num_gaussians))
        self.sigma = tf.Variable(tf.ones(shape=[1, num_gaussians]))

    def __call__(self, x):
        gaussians = tf.exp(-(((x - self.mu) / self.sigma) ** 2))
        return tf.squeeze(gaussians @ self.w + self.b)


def main(a):
    data = Data(FLAGS.num_samples, FLAGS.sigma_noise, FLAGS.random_seed)
    model = Model(FLAGS.num_gaussians)
    optimizer = tf.optimizers.SGD(learning_rate=FLAGS.learning_rate)

    for i in range(FLAGS.num_iter):
        with tf.GradientTape() as tape:
            x, y = data.get_batch(FLAGS.batch_size)
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
        data.x,
        data.y,
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
    app.run(main)
