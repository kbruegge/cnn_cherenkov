from cnn_cherenkov import image_io
from cnn_cherenkov import networks
from cnn_cherenkov import plotting
import click
import os


model_path = './data/model/supervised/fact.tflearn.index'
checkpoint_path = './data/model/supervised/'
predictions_path = './build/predictions_supervised_mc.hdf5'


def load_model(network):
    import tflearn
    model = tflearn.DNN(network,
                        checkpoint_path=checkpoint_path,
                        max_checkpoints=1,
                        tensorboard_verbose=3,
                        )

    if os.path.exists(model_path):
        print('Loading Model')
        model.load(model_path)

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




@cli.command()
@click.option('-e', '--epochs', default=1)
@click.option('-l', '--learning_rate', default=0.001)
@click.option('--mc/--data', default=True)
@click.option('-n', '--number_of_training_samples', default=100000)
@click.option('-b', '--batch_size', default=512)
@click.option('-o', '--optimizer', type=click.Choice(['adam', 'sgd']), default='adam')
@click.pass_context
def train(ctx, epochs, learning_rate, mc, number_of_training_samples, batch_size, optimizer):
    from tflearn.layers.estimator import regression
    network = ctx.obj['network']

    if mc:
        print('Loading simulated data.')
        X, Y = image_io.get_mc_training_data(end=number_of_training_samples)
    else:
        print('Loading Crab data.')
        df, images = image_io.load_crab_data(end=number_of_training_samples)
        _, X, Y = image_io.create_training_sample(df, images)

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
              snapshot_step=50,
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
