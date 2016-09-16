"""
Quick and dirty script to plot the loss.
"""
from __future__ import print_function

import re
import os.path
import argparse
from matplotlib import pyplot as plt


def build_parser():
    parser = argparse.ArgumentParser(__doc__)

    parser.add_argument(
        '-l', '--logfile',
        help='Path to Caffe logfile.',
        dest='log',
        required=True,
    )

    parser.add_argument(
        '-p', '--partition',
        dest='partition',
        action='append',
        choices=['test', 'train'],
        default=None
    )

    return parser


def main(args):
    assert os.path.isfile(args.log), 'File must exist'

    with open(args.log) as fd:
        log_data = fd.read()

    if 'train' in args.partition:
        iters = re.findall('Iteration (\d+), loss', log_data)
        iters = map(int, iters)

        losses = re.findall('loss = (\d+.?\d*)', log_data)
        iters = map(float, iters)

        plt.plot(iters, losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()

        accuracies = re.findall(
            'Train net output #0: Accuracy1 = (\d+.?\d*)',
            log_data
        )

        plt.plot(iters, accuracies)
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.show()

    if 'test' in args.partition:
        test_iters = re.findall('Iteration (\d+), Testing', log_data)
        test_iters = map(int, test_iters)

        test_acc = re.findall(
            'Test net output #0: Accuracy1 = (\d+.?\d*)',
            log_data
        )
        plt.plot(test_iters, test_acc)
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.show()


if __name__ == '__main__':
    args = build_parser().parse_args()
    main(args)
