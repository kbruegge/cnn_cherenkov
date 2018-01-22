from cnn_cherenkov import image_io
from cnn_cherenkov import networks
from cnn_cherenkov import plotting
import tflearn
import click
import os

@click.command()
@click.option('-s', '--start', default=0)
@click.option('-e', '--end', default=-1)
@click.option('-l', '--learning_rate', default=0.001)
@click.option('-o', '--operation', default='train', type=click.Choice(['train', 'apply', 'plot']))
@click.option('-n', '--network', default='alexnet')
@click.option('-p', '--epochs', default=1)
def main(start, end, learning_rate, operation, network, epochs):
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
                        tensorboard_verbose=3,
                        )

    if operation == 'train':
        X, Y = image_io.get_mc_training_data(start=start, end=end)

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
                  snapshot_step=50,
                  snapshot_epoch=True,
                  run_id='fact_tflearn'
                  )

        model.save(model_path)
    elif operation=='apply':
        print('Loading Model...')
        model.load(model_path)
        from IPython import embed; embed()
        df = networks.apply_to_data(model)
        print('Writing {} events to file...'.format(len(df)))
        df.to_hdf('./build/predictions_supervised_mc.hdf5', key='events')

    elif operation=='plot':
        import matplotlib.pyplot as plt
        model.load(model_path)
        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 20))
        plotting.display_convolutions(model, 'conv1', axis=ax1)
        plotting.display_convolutions(model, 'conv2', axis=ax2)
        plt.show()
if __name__ == '__main__':
    main()
