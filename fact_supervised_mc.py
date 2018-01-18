from cnn_cherenkov import image_io
from cnn_cherenkov import networks
import tflearn
import click
import os


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
        network = networks.alexnet(learning_rate=learning_rate)

    if network == 'simple':
        network = networks.simple(learning_rate=learning_rate)

    model = tflearn.DNN(network,
                        checkpoint_path='./data/model/supervised_data/',
                        max_checkpoints=1,
                        tensorboard_verbose=2,
                        )

    if train:
        df, images = image_io.load_crab_data(start, end)
        if os.path.exists('./data/model/supervised_data/fact.tflearn.index'):
            print('Loading Model')
            model.load('./data/model/supervised_data/fact.tflearn')

        _, X, Y = image_io.create_training_sample(df, images)
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

        model.save('./data/model/supervised_data/fact.tflearn')
    else:
        print('Loading Model...')
        model.load('./data/model/supervised_data/fact.tflearn')
        network.apply_to_data(model)
        print('Writing {} events to file...'.format(len(df)))
        df.to_hdf('./build/predictions.h5', key='events')


if __name__ == '__main__':
    main()
