
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import image_io
import numpy as np
from tqdm import tqdm
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import click


def scale_images(images):
    #images[images == 3] = 0
    qmax = np.percentile(images, q=99.5, axis=(1, 2))
    a = images / qmax[:, np.newaxis, np.newaxis]
    a = np.roll(a, 23)

    return a.reshape((len(images), 45, 46, -1))


def load_crab_training_data(start=0, end=-1):
    path = './data/crab_images.hdf5'
    df, images = image_io.read_n_rows(path, start=start, end=end)

    df['prediction_label'] = np.where(df.gamma_prediction > 0.8, 0, 1)

    X = scale_images(images)
    Y = df.prediction_label.values.astype(np.float32)

    N = len(df)
    ids_true = df[df.prediction_label == 1].index.values
    ids_true = np.random.choice(ids_true, N // 2)
    ids_false = df[df.prediction_label == 0].index.values
    ids_false = np.random.choice(ids_false, N // 2)
    ids = np.append(ids_false, ids_true)

    X = X[ids]
    Y = Y[ids]

    print('Loaded {} positive labels and {} negative labels'.format(
        np.sum(Y), N - np.sum(Y)))
    Y = OneHotEncoder().fit_transform(Y.reshape(-1, 1)).toarray()
    return df, X, Y


def load_crab_test_data(start=0, end=-1):
    path = './data/crab_images.hdf5'
    df, images = image_io.read_n_rows(path, start=start, end=end)
    X = scale_images(images)

    return df, X


def simple(learning_rate=0.001):
    print('Building simple net')
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
                         loss='binary_crossentropy',
                         learning_rate=learning_rate
                         )
    return network


def alexnet(learning_rate=0.001):
    print('Building AlexNet')
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
                         loss='binary_crossentropy',
                         learning_rate=learning_rate
                         )
    return network


@click.command()
@click.option('-s', '--start', default=0)
@click.option('-e', '--end', default=10000)
@click.option('-l', '--learning_rate', default=0.001)
@click.option('--train/--apply', default=True)
@click.option('-n', '--network', default='alexnet')
def main(start, end, learning_rate, train, network):
    # config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth = True))
    # session = tf.Session(config=config)

    if network == 'alexnet':
        network = alexnet(learning_rate=learning_rate)

    if network == 'simple':
        network = simple(learning_rate=learning_rate)

    model = tflearn.DNN(network,
                        checkpoint_path='./data/model/',
                        max_checkpoints=1,
                        tensorboard_verbose=2,
                        )

    if train:
        df, X, Y = load_crab_training_data(start, end)
        model.fit(X,
                  Y,
                  n_epoch=1,
                  validation_set=0.2,
                  shuffle=True,
                  show_metric=True,
                  batch_size=512,
                  snapshot_step=100,
                  snapshot_epoch=False,
                  run_id='fact_tflearn'
                  )

        model.save('./data/model/fact.tflearn')
    else:
        print('Loading Model')
        model.load('./data/model/fact.tflearn')
        df, X = load_crab_test_data(start, end)
        N = len(df)
        idx = np.array_split(np.arange(0, N), N / 128)

        print('Starting Batch Processing')
        predictions = []
        for ids in tqdm(idx):
            l = ids[0]
            u = ids[-1]
            batch = X[l:(u + 1)]
            predictions.extend(model.predict(batch)[:, 0])

        df['predictions_convnet'] = predictions
        df.to_hdf('./build/predictions.h5', key='events')


if __name__ == '__main__':
    main()
