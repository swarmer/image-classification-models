#!/usr/bin/env python
import argparse

import tensorflow as tf

import dataset


class SoftmaxRegressionModel:
    def __init__(self, learning_rate=0.5):
        self.lr = learning_rate

        self.x = tf.placeholder(tf.float32, (None, dataset.IMAGE_SIZE))
        self.y_expected = tf.placeholder(tf.float32, (None, dataset.LABEL_COUNT))

        self.w = tf.Variable(tf.zeros((dataset.IMAGE_SIZE, dataset.LABEL_COUNT)))
        self.b = tf.Variable(tf.zeros((dataset.LABEL_COUNT,)))

        self.y = tf.nn.softmax(tf.matmul(self.x, self.w) + self.b)
        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_expected, logits=self.y)
        )

        self.train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cross_entropy)

        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_expected, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def run_training_step(self, sess, x_array, y_expected_array):
        sess.run(self.train_step, feed_dict={
            self.x: x_array,
            self.y_expected: y_expected_array,
        })

    def evaluate(self, sess, x_array, y_expected_array):
        result = sess.run(self.accuracy, feed_dict={
            self.x: x_array,
            self.y_expected: y_expected_array,
        })
        return result

    def query(self, sess, x_array):
        result = sess.run(self.y, feed_dict={
            self.x: x_array,
        })
        return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-file', required=True, nargs='+',
        help='Path to a file with training data',
    )
    parser.add_argument(
        '--test-file', required=True, nargs='+',
        help='Path to a file with test data',
    )
    parser.add_argument(
        '--epochs', type=int, default=10,
        help='How many epochs to train for',
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.1,
        help='Learning rate',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    softmax_regression = SoftmaxRegressionModel()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        try:
            print('Training...')
            for epoch in range(1, args.epochs + 1):
                print(f'Epoch {epoch}')
                train_set = dataset.read_dataset(args.train_file, normalize=False)

                for images, labels in train_set.pairs:
                    softmax_regression.run_training_step(sess, images, labels)
        except KeyboardInterrupt:
            pass

        test_set = dataset.read_dataset(args.test_file, normalize=False)
        for images, labels in test_set.pairs:
            accuracy = softmax_regression.evaluate(sess, images, labels)
            print(f'Accuracy: {accuracy}')


if __name__ == '__main__':
    main()
