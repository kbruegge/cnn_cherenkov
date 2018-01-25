import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from cnn_cherenkov import image_io
from cnn_cherenkov import networks
from cnn_cherenkov import plotting
import click
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)


model_path = './data/model/supervised/fact.tflearn'
checkpoint_path = './data/model/supervised'
predictions_path = './build/predictions_supervised_mc.hdf5'


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
        return model
    else:
        return None



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
        df, X, Y = image_io.load_crab_training_data(N=number_of_training_samples)
        #df, images = load_crab_data(0, number_of_training_samples)
        #X, Y = sample_training_data(df, images)

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
@click.option('--data', '-d', default='crab', type=click.Choice(['crab', 'gamma', 'proton']))
@click.pass_context
def apply(ctx, data):
    network = ctx.obj['network']
    model = load_model(network)

    if not model:
        print('No model trained yet. Do so first.')
        return

    if data == 'crab':
        df = image_io.apply_to_observation_data(model)
    elif data == 'gamma':
        df = image_io.apply_to_mc(model, path='./data/gamma_images.hdf5')
    elif data == 'proton':
        df = image_io.apply_to_mc(model, path='./data/proton_images.hdf5')
    print('Writing {} events to file {}'.format(len(df), predictions_path))
    df.to_hdf(predictions_path, key='events')


if __name__ == '__main__':
    cli(obj={})
