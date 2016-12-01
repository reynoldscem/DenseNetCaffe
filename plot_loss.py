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

    parser.add_argument(
        '-n', '--network',
        dest='network_name',
        default=''
    )

    parser.add_argument(
        '-o', '--output',
        dest='output',
        default=None
    )

    parser.add_argument(
        '--no-show',
        dest='show',
        action='store_false',
        default=True
    )

    return parser


def main(args):
    assert os.path.isfile(args.log), 'File must exist'

    if args.output is not None:
        assert os.path.exists(args.output), 'Output dir must exist.'
        out_dir = os.path.abspath(args.output)

    with open(args.log) as fd:
        log_data = fd.read()

    if 'train' in args.partition:
        iters = re.findall('Iteration (\d+), loss', log_data)
        iters = map(int, iters)

        losses = re.findall('Iteration \d+, loss = (\d+.?\d*)', log_data)
        iters = map(float, iters)

        plt.figure()
        plt.plot(iters, losses)
        plt.title('Train loss for {}'.format(args.network_name))
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        if args.output:
            plt.savefig(os.path.join(out_dir, 'train_loss.png'))

        if args.show:
            plt.show()

        accuracies = re.findall(
            'Train net output #0: (?:A|a)ccuracy1? = (\d+.?\d*)',
            log_data
        )

        # If optimisation finished there is an extra match for iterations.
        plt.figure()
        plt.plot(iters[:len(accuracies)], accuracies)
        plt.title('Train accuracy for {}'.format(args.network_name))
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        if args.output:
            plt.savefig(os.path.join(out_dir, 'train_acc.png'))

        if args.show:
            plt.show()

    if 'test' in args.partition:
        test_iters = re.findall('Iteration (\d+), Testing', log_data)
        test_iters = map(int, test_iters)

        test_acc = re.findall(
            'Test net output #\d: (?:A|a)ccuracy1? = (\d+.?\d*)',
            log_data
        )
        test_loss = re.findall(
            'Test net output #\d: loss = (\d+.?\d*)',
            log_data
        )
        plt.figure()
        plt.plot(test_iters, test_acc)
        plt.title('Test accuracy for {}'.format(args.network_name))
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        if args.output:
            plt.savefig(os.path.join(out_dir, 'test_acc.png'))

        if args.show:
            plt.show()

        plt.figure()
        plt.plot(test_iters, test_loss)
        plt.title('Test loss for {}'.format(args.network_name))
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        if args.output:
            plt.savefig(os.path.join(out_dir, 'test_loss.png'))

        if args.show:
            plt.show()


if __name__ == '__main__':
    args = build_parser().parse_args()
    main(args)
