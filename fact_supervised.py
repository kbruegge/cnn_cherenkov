from cnn_cherenkov import image_io
from cnn_cherenkov import networks
from cnn_cherenkov import plotting
import click
import os
import numpy as np
import fact.io as fio
from sklearn.preprocessing import OneHotEncoder
import h5py
import pandas as pd


model_path = './data/model/supervised/fact.tflearn'
checkpoint_path = './data/model/supervised/fact'
predictions_path = './build/predictions_supervised_mc.hdf5'


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



def load_model(network):
    import tflearn
    model = tflearn.DNN(network,
                        checkpoint_path=checkpoint_path,
                        max_checkpoints=1,
                        tensorboard_verbose=3,
                        )

    p = '{}.index'.format(model_path)
    if os.path.exists(p):
        print('Loading Model')
        model.load(model_path)
    else:
        print('No model to load. Starting from scratch')
    return model


@click.group()
@click.option('--network', '-n', default='alexnet')
@click.pass_context
def cli(context, network):
    print('Loading {}'.format(network))

    if network == 'alexnet':
        network = networks.alexnet()

    if network == 'simple':
        network = networks.simple()

    context.obj['network'] = network


@cli.command()
@click.option('-o', '--output_path', default=None, type=click.Path(), help='If supplied will save the plot. Otherwise will call plt.show')
@click.option('-c', '--cmap', default='gray',)
@click.pass_context
def plot(ctx, output_path, cmap):
    '''
    Plots the weights of the first two convolutional layers in the network.
    '''
    network = ctx.obj['network']
    model = load_model(network)
    print('Plotting weights for {} '.format(model))
    import matplotlib.pyplot as plt
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 10))
    plotting.display_convolutions(model, 'conv1', axis=ax1, cmap=cmap)
    plotting.display_convolutions(model, 'conv2', axis=ax2, cmap=cmap)
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


@cli.command()
def clear():
    '''
    Clear all checkpoints and trained models
    '''
    click.confirm('Do you want to delete all pretrained models?', abort=True)
    import shutil
    shutil.rmtree(checkpoint_path)
    import os
    os.makedirs(checkpoint_path, exist_ok=True)



@cli.command()
@click.option('-e', '--epochs', default=1)
@click.option('-l', '--learning_rate', default=0.001)
@click.option('--mc/--data', default=True)
@click.option('-n', '--number_of_training_samples', default=100000)
@click.option('-b', '--batch_size', default=512)
@click.option('-o', '--optimizer', type=click.Choice(['adam', 'momentum', 'sgd']), default='adam')
@click.pass_context
def train(ctx, epochs, learning_rate, mc, number_of_training_samples, batch_size, optimizer):
    from tflearn.layers.estimator import regression
    network = ctx.obj['network']

    if mc:
        print('Loading simulated data.')
        X, Y = image_io.load_mc_training_data(N=number_of_training_samples)
    else:
        print('Loading Crab data.')
        #_, X, Y = image_io.load_crab_training_data(N=number_of_training_samples)
        df, images = load_crab_data(0, number_of_training_samples)
        X, Y = sample_training_data(df, images)

    network = regression(network,
                         optimizer=optimizer,
                         loss='binary_crossentropy',
                         learning_rate=learning_rate
                         )

    model = load_model(network)
    model.fit(X,
              Y,
              n_epoch=epochs,
              validation_set=0.2,
              shuffle=True,
              show_metric=True,
              batch_size=batch_size,
              snapshot_step=25,
              snapshot_epoch=True,
              run_id='fact_tflearn'
              )

    model.save(model_path)


@cli.command()
@click.pass_context
def apply(ctx):
    network = ctx.obj['network']
    model = load_model(network)
    df = networks.apply_to_data(model)
    print('Writing {} events to file {}'.format(len(df), predictions_path))
    df.to_hdf(predictions_path, key='events')


if __name__ == '__main__':
    cli(obj={})
