import click
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


@click.command()
@click.argument('predictions', type=click.Path(exists=True, dir_okay=False))
@click.option('--net/--no-net', default=True)
def main(predictions, net):
    df = pd.read_hdf(predictions)
    if net:
        print('using net predictions')
        d = df.predictions_convnet
    else:
        print('using standart predictions')
        d = df.gamma_prediction
    plt.hist(d, bins=np.linspace(0, 1, 50))
    plt.show()


if __name__ == '__main__':
    main()
