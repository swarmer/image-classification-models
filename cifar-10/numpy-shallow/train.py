#!/usr/bin/env python
import argparse
import itertools
import pickle
import random

import numpy as np

import dataset
import network


random.seed(7)


IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
INPUT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH
OUTPUT_SIZE = 10


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
        '--train-limit', type=int,
        help='How many examples to pick from train dataset',
    )
    parser.add_argument(
        '--test-limit', type=int,
        help='How many examples to pick from test dataset',
    )
    parser.add_argument(
        '--hidden-count', type=int, default=100,
        help='Hidden node count',
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

    nn = network.NeuralNetwork(
        input_count=dataset.IMAGE_SIZE, hidden_count=args.hidden_count, output_count=dataset.LABEL_COUNT,
        learning_rate=args.learning_rate,
    )

    try:
        print('Training...')
        for epoch in range(1, args.epochs + 1):
            print(f'Epoch {epoch}')
            train_set = dataset.read_dataset(args.train_file)
            nn.train(itertools.islice(train_set.pairs, 0, args.train_limit))
    except KeyboardInterrupt:
        pass

    print('Testing...')
    test_set = dataset.read_dataset(args.test_file)
    results = []

    for img, label in itertools.islice(test_set.pairs, 0, args.test_limit):
        output = nn.query(img)
        output_label = np.argmax(output)

        results.append(output_label == np.argmax(label))

    print(f'Test set precision: {np.mean(results)}')

    print('Dumping neural net...')
    with open('model.bin', 'wb') as model_file:
        pickle.dump(nn, model_file)


if __name__ == '__main__':
    main()
