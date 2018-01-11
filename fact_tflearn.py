
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import image_io
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import click
#import tensorflow as tf


def scale_images(images):
    X = images.astype(np.float32).reshape(len(images), -1)
    return StandardScaler().fit_transform(X).reshape((len(images), 45, 46, -1))

def load_data(start=0, end=-1):
    path = './data/crab_images.hdf5'
    df, images = image_io.read_n_rows(path, start=start, end=end)

    X = scale_images(images)
    #Y = np.digitize(df.gamma_prediction.values.astype(np.float32), bins=np.linspace(0, 1, 2, endpoint=False)) - 1
    Y =  np.digitize(df.gamma_prediction.values.astype(np.float32), bins=np.linspace(0, 1, 10)) - 1
    #from IPython import embed; embed()
    N = len(df)
    ids_true = np.random.choice(np.where(Y>=7)[0], N//2)
    ids_false = np.random.choice(np.where(Y<7)[0], N//2)
    ids = np.append(ids_false, ids_true)

    df = df.iloc[ids]
    Y = Y[ids]
    X = X[ids]
    print('Loaded {} positive labels and {} negative labels'.format(np.sum(Y), N - np.sum(Y)))
    Y = OneHotEncoder().fit_transform(Y.reshape(-1, 1)).toarray()
    return df, X, Y

def alexnet(learning_rate=0.001):
    # Building 'AlexNet'
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
    network = regression(network, optimizer='momentum',
                                 loss='binary_crossentropy',
                                 learning_rate=learning_rate)
    return network


@click.command()
@click.option('-s', '--start', default=0)
@click.option('-e', '--end', default=10000)
@click.option('-l', '--learning_rate', default=0.001)
@click.option('--train/--apply', default=True)
def main(start, end, learning_rate,  train):
    #config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth = True))
    #session = tf.Session(config=config)
    network = alexnet(learning_rate=learning_rate)
    model = tflearn.DNN(network, checkpoint_path='./data/model_alexnet',
                                max_checkpoints=1, tensorboard_verbose=2,)

    if train:
        df, X, Y = load_data(start, end)
        model.fit(X, Y, n_epoch=1, validation_set=0.1, shuffle=True,
                          show_metric=True, batch_size=256, snapshot_step=200,
                          snapshot_epoch=False, run_id='fact_tflearn')

        model.save('./data/model_alexnet/fact_tflearn')
    else:

        df, X, Y = load_data(start, end)
        model.load('./data/model_alexnet/fact_tflearn')
        predictions = model.predict(X)
        df['predictions_convnet'] = predictions[:, 0]
        df.to_hdf('./build/predictions.h5', key='events')


if __name__=='__main__':
    main()
