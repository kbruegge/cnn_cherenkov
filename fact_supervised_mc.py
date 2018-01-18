from cnn_cherenkov import image_io
from cnn_cherenkov import networks
import tflearn
import click
import os
import pandas as pd
import h5py
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle



def read_rows(path, start=0, end=1000):
    '''
    read given rows from carb images.
    return dataframe containg high level infor and iimages (df, images)
    '''
    f = h5py.File(path)
    night = f['events/night'][start:end]
    run = f['events/run_id'][start:end]
    event = f['events/event_num'][start:end]
    az = f['events/az'][start:end]
    zd = f['events/zd'][start:end]

    df = pd.DataFrame({'night': night, 'run_id': run, 'event_num': event, 'zd': zd, 'az': az, })
    return df, f['events/image'][start:end]




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
    checkpoint_path = './data/model/supervised_mc/'
    model_path = './data/model/supervised_mc/fact.tflearn.index'

    if network == 'alexnet':
        network = networks.alexnet(learning_rate=learning_rate)

    if network == 'simple':
        network = networks.simple(learning_rate=learning_rate)

    model = tflearn.DNN(network,
                        checkpoint_path=checkpoint_path,
                        max_checkpoints=1,
                        tensorboard_verbose=2,
                        )

    if train:
        df_gammas, images_gammas = read_rows('./data/gamma_images.hdf5', start=start, end=end)
        df_protons, images_protons = read_rows('./data/proton_images.hdf5', start=start, end=end)

        images_gammas = np.transpose(images_gammas, axes=[0, 2, 1])
        images_protons = np.transpose(images_protons, axes=[0, 2, 1])

        images_gammas = image_io.scale_images(images_gammas)
        images_protons = image_io.scale_images(images_protons)
        X = np.vstack([images_gammas, images_protons])

        Y = np.append(np.ones(len(df_gammas)), np.zeros(len(df_protons)))
        Y = OneHotEncoder().fit_transform(Y.reshape(-1, 1)).toarray()

        X, Y = shuffle(X, Y)

        # import IPython; IPython.embed()


        if os.path.exists('{}.index'.format(model_path)):
            print('Loading Model')
            model.load(model_path)

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

        model.save(model_path)
    else:
        print('Loading Model...')
        model.load(model_path)
        df = network.apply_to_data(model)
        print('Writing {} events to file...'.format(len(df)))
        df.to_hdf('./build/predictions_supervised_mc.hdf5', key='events')


if __name__ == '__main__':
    main()
