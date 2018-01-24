
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import numpy as np
from tqdm import tqdm
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import click
import fact.io as fio
import pandas as pd
import os
import h5py



def read_n_rows(path, start=0, end=1000):
    f = h5py.File(path)
    night = f['events/night'][start:end]
    run = f['events/run_id'][start:end]
    event = f['events/event_num'][start:end]
    az = f['events/az'][start:end]
    zd = f['events/zd'][start:end]

    df = pd.DataFrame({'night': night, 'run_id': run, 'event_num': event, 'zd': zd, 'az': az,})
    return df, f['events/image'][start:end]


def number_of_images(path):
    f = h5py.File(path)
    return len(f['events/night'])


def scale_images(images):
    images[images < 3] = 0
    qmax = np.percentile(images, q=99.5, axis=(1, 2))
    a = images / qmax[:, np.newaxis, np.newaxis]
    return a.reshape((len(images), 45, 46, -1))


def sample_training_data(df, images):
    df = df.reset_index()

    Y = df.prediction_label.values.astype(np.float32)

    N = len(df)
    ids_true = df[df.prediction_label == 1].index.values
    ids_true = np.random.choice(ids_true, N // 2)
    ids_false = df[df.prediction_label == 0].index.values
    ids_false = np.random.choice(ids_false, N // 2)
    ids = np.append(ids_false, ids_true)

    X = images[ids]
    Y = Y[ids]

    print('Loaded {} positive labels and {} negative labels'.format(
        np.sum(Y), N - np.sum(Y)))
    Y = OneHotEncoder().fit_transform(Y.reshape(-1, 1)).toarray()
    return X, Y


def load_crab_data(start=0, end=-1):
    df, images = read_n_rows('./data/crab_images.hdf5', start=start, end=end)
    dl3 = fio.read_data('./data/dl3/open_crab_sample_dl3.hdf5', key='events')
    dl3 = dl3.set_index(['night', 'run_id', 'event_num'])

    df['int_index'] = df.index
    df = df.set_index(['night', 'run_id', 'event_num'])


    data = df.join(dl3, how='inner')
    print('Events in open data sample: {}, events in photons_stream: {}, events in joined data: {}'.format(len(dl3), len(df), len(data)))

    if len(data) == 0:
        return [], []

    data['prediction_label'] = np.where(data.gamma_prediction > 0.8, 0, 1)

    images = scale_images(images[data.int_index])

    assert len(images) == len(data)

    return data, images


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
@click.option('-e', '--end', default=-1)
@click.option('-l', '--learning_rate', default=0.001)
@click.option('--train/--apply', default=True)
@click.option('-n', '--network', default='alexnet')
@click.option('-p', '--epochs', default=1)
def main(start, end, learning_rate, train, network, epochs):
    # config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth = True))
    # session = tf.Session(config=config)

    if network == 'alexnet':
        network = alexnet(learning_rate=learning_rate)

    if network == 'simple':
        network = simple(learning_rate=learning_rate)

    model = tflearn.DNN(network,
                        checkpoint_path='./data/model/old_process',
                        max_checkpoints=1,
                        tensorboard_verbose=2,
                        )

    if train:
        df, images = load_crab_data(start, end)
        if os.path.exists('./data/model/old_process/fact.tflearn.index'):
            print('Loading Model')
            model.load('./data/model/old_process/fact.tflearn')

        X, Y = sample_training_data(df, images)
        model.fit(X,
                  Y,
                  n_epoch=epochs,
                  validation_set=0.2,
                  shuffle=True,
                  show_metric=True,
                  batch_size=512,
                  snapshot_step=25,
                  snapshot_epoch=True,
                  run_id='fact_tflearn'
                  )

        model.save('./data/model/old_process/fact.tflearn')
    else:
        print('Loading Model')
        model.load('./data/model/old_process/fact.tflearn')
        N = number_of_images('./data/crab_images.hdf5')
        idx = np.array_split(np.arange(0, N), N / 8000)
        dfs = []
        event_counter = 0
        for ids in tqdm(idx):
            l = ids[0]
            u = ids[-1]
            df, images = load_crab_data(l, u+1)
            event_counter += len(df)
            if len(df) == 0:
                continue
            predictions = model.predict(images)[:, 0]

            df['predictions_convnet'] = predictions
            dfs.append(df)

        print('Concatenating {} data frames'.format(len(dfs)))
        df = pd.concat(dfs)
        assert event_counter == len(df)
        print('Writing {} events to file'.format(event_counter))
        df.to_hdf('./build/predictions_old_process.h5', key='events')


if __name__ == '__main__':
    main()
