import click
import pandas as pd
import matplotlib.pyplot as plt


@click.command()
@click.argument('predictions', type=click.Path(exists=True, dir_okay=False))
@click.option('-t', '--threshold', default=0.5)
@click.option('--net/--no-net', default=True)
def main(predictions, threshold, net):
    df = pd.read_hdf(predictions)
    if net:
        df = df.query('predictions_convnet > {}'.format(threshold))
    else:
        df = df.query('gamma_prediction > {}'.format(threshold))
    plt.hist2d(df.ra_prediction, df.dec_prediction, bins=40)
    plt.colorbar()
    if net:
        plt.title('net predictions')
    else:
        plt.title('rf predictions')
    plt.show()


if __name__ == '__main__':
    main()
