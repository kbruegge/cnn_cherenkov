from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from . import image_io
import numpy as np
from tqdm import tqdm
import pandas as pd


def simple(learning_rate=0.001, loss=None):
    if not loss:
        loss = 'binary_crossentropy'

    print('Building Simple Net with loss {}'.format(loss))

    network = input_data(shape=[None, 46, 45, 1])

    network = conv_2d(network, 8, 64, activation='relu', name='conv1')
    network = max_pool_2d(network, 3, strides=1)

    network = conv_2d(network, 4, 32, activation='relu', name='conv2')
    network = max_pool_2d(network, 3, strides=1)


    network = fully_connected(network, 100, activation='relu', name='fc1')
    network = dropout(network, 0.5)
    network = fully_connected(network, 100, activation='relu', name='fc2')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax', name='fc3')
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
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network,
                         optimizer='adam',
                         loss=loss,
                         learning_rate=learning_rate
                         )
    return network


def alexnet_region(loss, learning_rate=0.001):
    print('Building AlexNet with loss {}'.format(loss))
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
    network = regression(network,
                         optimizer='momentum',
                         loss=loss,
                         learning_rate=learning_rate
                         )
    return network


def apply_to_data(model):
    N = image_io.number_of_images('./data/crab_images.hdf5')
    idx = np.array_split(np.arange(0, N), N / 8000)
    dfs = []
    event_counter = 0
    try:
        for ids in tqdm(idx):
            l = ids[0]
            u = ids[-1]
            df, images = image_io.load_crab_data(l, u + 1)
            event_counter += len(df)
            if len(df) == 0:
                continue
            predictions = model.predict(images)[:, 0]

            df['predictions_convnet'] = predictions
            dfs.append(df)
    except KeyboardInterrupt:
        print('Stopping process..')
    finally:
        print('Concatenating {} data frames'.format(len(dfs)))
        df = pd.concat(dfs)
        assert event_counter == len(df)
        return df
