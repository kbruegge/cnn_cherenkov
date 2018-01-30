import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from cnn_cherenkov import image_io
from cnn_cherenkov import networks
from cnn_cherenkov import plotting
import click
import tensorflow as tf
from tensorflow.python.framework.errors_impl import NotFoundError


tf.logging.set_verbosity(tf.logging.ERROR)


model_path = './data/model/supervised/fact.tfl'
checkpoint_path = './data/model/supervised/fact.tfl.ckpt'


def load_model(network):
    import tflearn
    model = tflearn.DNN(network,
                        checkpoint_path=checkpoint_path,
                        tensorboard_verbose=3,
                        )

    p = '{}.index'.format(model_path)
    if os.path.exists(p):
        print('Loading Model')
        try:
            model.load(model_path)
        except NotFoundError as e:
            print('Loading only weights.')
            model.load(model_path, weights_only=True)
    return model



@click.group()
@click.option('--network', '-n', default='alexnet')
@click.pass_context
def cli(context, network):
    # print('Loading {}'.format(network))

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
    import os
    directory = os.path.dirname(model_path)
    for f in os.listdir(directory):
        os.remove(os.path.join(directory, f))


@cli.command()
@click.option('-e', '--epochs', default=1)
@click.option('-l', '--learning_rate', default=0.001)
@click.option('-d', '--data_source', default='mc', type=click.Choice(['simulation', 'mc', 'data', 'observations', 'mix', 'both']))
@click.option('-n', '--number_of_training_samples', default=100000)
@click.option('-b', '--batch_size', default=512)
@click.option('-o', '--optimizer', type=click.Choice(['adam', 'momentum', 'sgd']), default='adam')
@click.pass_context
def train(ctx, epochs, learning_rate, data_source, number_of_training_samples, batch_size, optimizer):
    from tflearn.layers.estimator import regression
    network = ctx.obj['network']

    if data_source in ['simulation', 'mc']:
        print('Loading simulated data.')
        X, Y = image_io.load_mc_training_data(N=number_of_training_samples)
    elif data_source in ['observations', 'data']:
        print('Loading Crab data.')
        df, X, Y = image_io.load_crab_training_data(N=number_of_training_samples)
    elif data_source in ['mix', 'both']:
        print('Mixing observations and MC data.')
        X, Y = image_io.load_mc_data_mix(N=number_of_training_samples)


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
@click.argument('out_file', type=click.Path(dir_okay=False))
@click.option('--data', '-d', default='crab', type=click.Choice(['crab', 'gamma', 'proton']))
@click.option('--number_of_images', '-n', default=-1)
@click.pass_context
def apply(ctx, out_file, data, number_of_images):
    import fact.io as fio
    network = ctx.obj['network']
    model = load_model(network)

    p = '{}.index'.format(model_path)
    if not os.path.exists(p):
        print('No model trained yet. Do so first.')
        return

    if data == 'crab':
        df = image_io.apply_to_observation_data(model)

    elif data == 'gamma':
        df = image_io.apply_to_mc(model, path='./data/gamma_images.hdf5', N=number_of_images)
        shower_truth = fio.read_data('./data/gamma_images.hdf5', key='showers')
        fio.write_data(shower_truth, file_path=out_file, key='showers', use_hp5y=True)

    elif data == 'proton':
        df = image_io.apply_to_mc(model, path='./data/proton_images.hdf5', N=number_of_images)
        shower_truth = fio.read_data('./data/proton_images.hdf5', key='showers')
        fio.write_data(shower_truth, file_path=out_file, key='showers', use_hp5y=True)

    print('Writing {} events to file {}'.format(len(df), out_file))
    fio.write_data(df, out_file, key='events')


if __name__ == '__main__':
    cli(obj={})
