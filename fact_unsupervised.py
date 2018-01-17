import image_io
import networks
import tflearn
import click
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


def sigma_loss(y_pred, y_true):
    with tf.name_scope(None):
        on_region_radius_degree = 0.2
        alpha = 0.2

        theta = y_true
        theta_on = theta[:, 0]

        theta_off_1 = theta[:, 1]
        theta_off_2 = theta[:, 2]
        theta_off_3 = theta[:, 3]
        theta_off_4 = theta[:, 4]
        theta_off_5 = theta[:, 5]
        #
        zeros = tf.zeros(tf.shape(y_pred))[:, 0]
        N_on = tf.reduce_sum(tf.where(theta_on < on_region_radius_degree, y_pred[:, 0] * theta_on, zeros))
        N_off_1 = tf.reduce_sum(tf.where(theta_off_1 < on_region_radius_degree, (1 - y_pred[:, 0]) * theta_off_1, zeros))
        # N_off_1 = tf.reduce_sum(tf.where(theta_off_1 < on_region_radius_degree, (1 - y_pred[:, 0]) * theta_off_1))
        N_off_2 = tf.reduce_sum(tf.where(theta_off_2 < on_region_radius_degree, (1 - y_pred[:, 0]) * theta_off_2, zeros))
        N_off_3 = tf.reduce_sum(tf.where(theta_off_3 < on_region_radius_degree, (1 - y_pred[:, 0]) * theta_off_3, zeros))
        N_off_4 = tf.reduce_sum(tf.where(theta_off_4 < on_region_radius_degree, (1 - y_pred[:, 0]) * theta_off_4, zeros))
        N_off_5 = tf.reduce_sum(tf.where(theta_off_5 < on_region_radius_degree, (1 - y_pred[:, 0]) * theta_off_5, zeros))
        #
        N_off = N_off_1 + N_off_2 + N_off_3 + N_off_4 + N_off_5
        #
        S = (N_on - alpha * N_off) / tf.sqrt(N_on + alpha**2 * N_off)
        loss = 1 - S / tf.sqrt(tf.cast(tf.shape(theta_on)[0], tf.float32))
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

    network = networks.alexnet_region(sigma_loss, learning_rate=learning_rate)


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
                  batch_size=1024,
                  snapshot_step=100,
                  snapshot_epoch=False,
                  run_id='fact_tflearn'
                  )

        model.save('./data/model/fact.tflearn')
    else:
        print('Loading Model')
        model.load('./data/model/fact.tflearn')
        df = network.apply_to_data(model)
        print('Writing {} events to file'.format(len(df)))
        df.to_hdf('./build/predictions.h5', key='events')


if __name__ == '__main__':
    main()
