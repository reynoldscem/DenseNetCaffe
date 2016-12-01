from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import os.path

caffe_root = '/home/reynoldscem/caffe'


def bn_relu_conv(bottom, kernel_size, nout, stride, pad, dropout, dilation=1):
    # batch_norm = L.BatchNorm(
    #     bottom, in_place=False,
    #     param=[
    #         dict(lr_mult=0, decay_mult=0),
    #         dict(lr_mult=0, decay_mult=0),
    #         dict(lr_mult=0, decay_mult=0)
    #     ]
    # )
    # scale = L.Scale(
    #     batch_norm, bias_term=True, in_place=True,
    #     filler=dict(value=1), bias_filler=dict(value=0)
    # )
    # relu = L.ReLU(scale, in_place=True)
    relu = L.ReLU(bottom, in_place=True)
    conv = L.Convolution(
        relu, kernel_size=kernel_size, stride=stride,
        num_output=nout, pad=pad, bias_term=False, dilation=dilation,
        weight_filler=dict(type='msra'), bias_filler=dict(type='constant')
    )
    if dropout > 0:
        conv = L.Dropout(conv, dropout_ratio=dropout)
    return conv


def add_layer(bottom, num_filter, dropout, pad=1, dilation=1):
    conv = bn_relu_conv(
        bottom, kernel_size=3, nout=num_filter,
        stride=1, pad=pad, dropout=dropout, dilation=dilation
    )
    concate = L.Concat(bottom, conv, axis=1)
    return concate


def transition(bottom, num_filter, dropout):
    conv = bn_relu_conv(
        bottom, kernel_size=1, nout=num_filter,
        stride=1, pad=0, dropout=dropout
    )
    # pooling = L.Pooling(conv, pool=P.Pooling.AVE, kernel_size=2, stride=2)
    return conv


# change the line below to experiment with different setting
# depth -- must be 3n+4
# first_output -- #channels before entering the first dense block,
#   set it to be comparable to growth_rate
# growth_rate -- growth rate
# dropout -- set to 0 to disable dropout, non-zero number to set dropout rate
def densenet(
    data_file, mode='train', batch_size=64, depth=10,
    first_output=16, growth_rate=10, dropout=0.2,
    dense_blocks=2, split='trainval', tops=['color', 'label']
):
    # mean_file = os.path.join(caffe_root, 'examples/cifar10/mean.binaryproto')
    # data, label = L.Data(
    #     source=data_file, backend=P.Data.LMDB,
    #     batch_size=batch_size, ntop=2,
    #     transform_param=dict(
    #         mean_file=mean_file
    #     )
    # )

    data, label = L.Python(
        module='nyud_layers',
        layer='NYUDSegDataLayer', ntop=2,
        param_str=str(dict(
            nyud_dir='/media/reynoldscem/DATA/temp-datasets/nyud_gupta_new',
            split=split,
            tops=tops, seed=1337
        )))

    nchannels = first_output
    model = L.Convolution(
        data, kernel_size=3, stride=1, num_output=nchannels,
        pad=1, bias_term=False, weight_filler=dict(type='msra'),
        bias_filler=dict(type='constant')
    )
    if dropout > 0:
        model = L.Dropout(model, dropout_ratio=dropout)

    # 1 initial conv layer, a transition layer
    # between each pair of dense blocks,
    # and an inner product layer for classification.
    aux_layers = 1 + 1 + (dense_blocks - 1)
    N = (depth - aux_layers) / dense_blocks
    assert float(N).is_integer(), \
        'Depth != (depth - auxiliary layers)  / num_blocks'
    dilation = dict(zip(range(N), [1, 1, 1, 2, 2, 2, 4, 4, 8, 8, 16, 16]))
    # dilation = dict(zip(range(N), [1, 1, 2, 2, 2, 4, 4, 8, 8, 16, 16, 16]))
    dilation = dict(zip(range(N), [1, 2, 4, 8]))
    # dilation = dict(zip(range(N), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    for block in range(dense_blocks):
        for i in range(N):
            model = add_layer(
                model, growth_rate, dropout,
                pad=dilation[i], dilation=dilation[i]
            )
            nchannels += growth_rate
        if block < dense_blocks - 1:
            model = transition(model, nchannels, dropout)

    # model = L.BatchNorm(
    #     model, in_place=False,
    #     param=[
    #         dict(lr_mult=0, decay_mult=0),
    #         dict(lr_mult=0, decay_mult=0),
    #         dict(lr_mult=0, decay_mult=0)
    #     ]
    # )
    # model = L.Scale(
    #     model, bias_term=True, in_place=True,
    #     filler=dict(value=1), bias_filler=dict(value=0)
    # )
    # model = L.ReLU(model, in_place=True)
    model = L.ReLU(model, in_place=True)

    model = transition(model, 40, 0.)
    # model = L.Pooling(model, pool=P.Pooling.AVE, global_pooling=True)
    # model = L.InnerProduct(
    #     model, num_output=10, bias_term=True,
    #     weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
    # )
    loss = L.SoftmaxWithLoss(model, label)
    accuracy = L.Accuracy(model, label)
    return to_proto(loss, accuracy)


def make_net():
    # lmbd_path = 'examples/cifar10/cifar10_{}_lmdb'
    # train_path = os.path.join(caffe_root, lmbd_path.format('train'))
    # test_path = os.path.join(caffe_root, lmbd_path.format('test'))
    # change the path to your data. If it's not lmdb format,
    # also change first line of densenet() function.
    with open('train_densenet.prototxt', 'w') as f:
        print(
            str(densenet(None, batch_size=1, split='trainval')),
            file=f
        )

    with open('test_densenet.prototxt', 'w') as f:
        print(str(densenet(None, batch_size=1, split='test')), file=f)


def make_solver():
    s = caffe_pb2.SolverParameter()
    s.random_seed = 0xCAFFE

    s.train_net = 'train_densenet.prototxt'
    s.test_net.append('test_densenet.prototxt')
    s.test_interval = 800
    s.test_iter.append(200)

    s.max_iter = 100000
    s.type = 'Nesterov'
    s.display = 10

    s.base_lr = 0.1
    s.momentum = 0.9
    s.weight_decay = 1e-4

    s.lr_policy = 'multistep'
    s.gamma = 0.1
    s.stepvalue.append(int(0.5 * s.max_iter))
    s.stepvalue.append(int(0.75 * s.max_iter))
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    s.snapshot = 10000
    s.snapshot_prefix = 'snapshots/DenseNet_nyud'

    solver_path = 'solver.prototxt'
    with open(solver_path, 'w') as f:
        f.write(str(s))

if __name__ == '__main__':
    make_net()
    make_solver()
