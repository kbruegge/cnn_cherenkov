import image_io
import networks
import tflearn
import numpy as np
from tqdm import tqdm
import click
import pandas as pd
import os
import tensorflow as tf
from sklearn import preprocessing


theta_keys = [
    'theta_deg',
    'theta_deg_off_1',
    'theta_deg_off_2',
    'theta_deg_off_3',
    'theta_deg_off_4',
    'theta_deg_off_4'
]


def region_loss(y_pred, y_true):
    with tf.name_scope(None):
        # N = tf.size(y_pred)
        # print(y_true.shape)
        # print(y_pred.shape)
        # print(N)
        # theta = tf.reshape(y_true, [512, 6])
        # print(theta.shape)
        # distance_on = y_true[:, 0]
        theta = y_true
        a = y_pred[:, 0] * theta[:, 0]
        b =  (1 - y_pred[:, 0]) * theta[:, 1]
        c = (1 - y_pred[:, 0]) * theta[:, 2]
        d = (1 - y_pred[:, 0]) * theta[:, 3]
        e = (1 - y_pred[:, 0]) * theta[:, 4]
        f = (1 - y_pred[:, 0]) * theta[:, 5]
        # d = y_pred[:, 1] * theta[:, 3]
        loss = tf.reduce_mean(a + b + c + d + e + f)
        return loss


@click.command()
@click.option('-s', '--start', default=0)
@click.option('-e', '--end', default=-1)
@click.option('-l', '--learning_rate', default=0.001)
@click.option('--train/--apply', default=True)
@click.option('-n', '--network', default='alexnet')
@click.option('-p', '--epochs', default=1)
@click.option('--restore/--no-restore', default=True)
def main(start, end, learning_rate, train, network, epochs, restore):

    network = networks.alexnet_region(region_loss, learning_rate=learning_rate)


    model = tflearn.DNN(network,
                        checkpoint_path='./data/model/',
                        max_checkpoints=1,
                        tensorboard_verbose=2,
                        )

    if train:
        df, images = image_io.load_crab_data(start, end)
        if restore and os.path.exists('./data/model/fact.tflearn.index'):
            print('Loading Model')
            model.load('./data/model/fact.tflearn')

        df, X, gamma_label = image_io.create_training_sample(df, images)

        Y = df[theta_keys].values
        min_max_scaler = preprocessing.MinMaxScaler()
        Y = min_max_scaler.fit_transform(Y)
        print(Y.shape)
        model.fit(X,
                  Y,
                  n_epoch=epochs,
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
            print('stopping execution')
        finally:
            print('Concatenating {} data frames'.format(len(dfs)))
            df = pd.concat(dfs)
            assert event_counter == len(df)
            print('Writing {} events to file'.format(event_counter))
            df.to_hdf('./build/predictions.h5', key='events')


if __name__ == '__main__':
    main()
