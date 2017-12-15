import click
import pandas as pd
import matplotlib.pyplot as plt


@click.command()
@click.argument('predictions', type=click.Path(exists=True, dir_okay=False))
@click.option('-t', '--threshold', default=0.5)
def main(predictions, threshold):
    df = pd.read_hdf(predictions)
    df = df.query('predictions_convnet > {}'.format(threshold))
    plt.hist2d(df.ra_prediction, df.dec_prediction, bins=20)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
