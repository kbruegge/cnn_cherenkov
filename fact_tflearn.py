import image_io
import networks
import tflearn
import numpy as np
from tqdm import tqdm
import click
import pandas as pd
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
                        checkpoint_path='./data/model/',
                        max_checkpoints=1,
                        tensorboard_verbose=2,
                        )

    if train:
        df, images = image_io.load_crab_data(start, end)
        if os.path.exists('./data/model/fact.tflearn.index'):
            print('Loading Model')
            model.load('./data/model/fact.tflearn')

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

        model.save('./data/model/fact.tflearn')
    else:
        print('Loading Model')
        model.load('./data/model/fact.tflearn')
        N = image_io.number_of_images('./data/crab_images.hdf5')
        idx = np.array_split(np.arange(0, N), N / 8000)
        dfs = []
        event_counter = 0
        for ids in tqdm(idx):
            l = ids[0]
            u = ids[-1]
            df, images = image_io.load_crab_data(l, u+1)
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
        df.to_hdf('./build/predictions.h5', key='events')


if __name__ == '__main__':
    main()
