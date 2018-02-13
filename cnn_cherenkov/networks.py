from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization


def simple(learning_rate=0.001, loss=None):
    network = input_data(shape=[None, 46, 45, 1])

    network = conv_2d(network, 32, 32, strides=2,  activation='relu', name='conv1')
    network = max_pool_2d(network, 2,)
    network = local_response_normalization(network)

    network = conv_2d(network, 16, 16, activation='relu', name='conv2')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)

    network = conv_2d(network, 16, 16, activation='relu', name='conv3')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 32, 2, activation='relu', name='conv4')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)


    network = fully_connected(network, 300, activation='relu', name='fc1')
    network = dropout(network, 0.8)
    network = fully_connected(network, 300, activation='relu', name='fc2')
    network = dropout(network, 0.8)
    network = fully_connected(network, 300, activation='relu', name='fc3')
    network = dropout(network, 0.8)
    network = fully_connected(network, 300, activation='relu', name='fc4')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax', name='fc5', restore=False)
    return network



def simple_small(learning_rate=0.001, loss=None):
    network = input_data(shape=[None, 46, 45, 1])

    network = conv_2d(network, 8, 32, activation='relu', name='conv1')
    network = max_pool_2d(network, 3, strides=2)

    network = conv_2d(network, 4, 16, activation='relu', name='conv2')
    network = max_pool_2d(network, 3, strides=2)


    network = fully_connected(network, 100, activation='relu', name='fc1')
    network = dropout(network, 0.5)
    network = fully_connected(network, 100, activation='relu', name='fc2')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax', name='fc3', restore=True)
    return network


def alexnet(learning_rate=0.001, loss=None):
    network = input_data(shape=[None, 46, 45, 1])
    network = conv_2d(network, 96, 11, strides=4, activation='relu', name='conv1')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu', name='conv2')
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
    network = fully_connected(network, 2, activation='softmax', restore=True)
    return network


def alexnet_region(loss, learning_rate=0.001):
    network = input_data(shape=[None, 46, 45, 1])
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
    network = fully_connected(network, 6, activation='softmax')
    return network
