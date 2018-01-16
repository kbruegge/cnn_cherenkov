from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


def simple(learning_rate=0.001, loss=None):
    if not loss:
        loss = 'binary_crossentropy'

    print('Building Simple Net with loss {}'.format(loss))

    network = input_data(shape=[None, 45, 46, 1])

    network = conv_2d(network, 5, 10, activation='relu')
    network = max_pool_2d(network, 3, strides=1)

    network = conv_2d(network, 3, 10, activation='relu')
    network = max_pool_2d(network, 3, strides=1)

    network = conv_2d(network, 5, 10, activation='relu')
    network = max_pool_2d(network, 3, strides=1)

    network = conv_2d(network, 3, 10, activation='relu')
    network = max_pool_2d(network, 3, strides=1)

    network = fully_connected(network, 100, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 100, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 100, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network,
                         optimizer='adam',
                         loss=loss,
                         learning_rate=learning_rate
                         )
    return network


def alexnet(learning_rate=0.001, loss=None):
    if not loss:
        loss = 'binary_crossentropy'

    print('Building AlexNet with loss {}'.format(loss))
    network = input_data(shape=[None, 45, 46, 1])
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network,
                         optimizer='momentum',
                         loss=loss,
                         learning_rate=learning_rate
                         )
    return network
