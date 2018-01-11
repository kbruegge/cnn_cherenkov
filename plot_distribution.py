import click
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


@click.command()
@click.argument('predictions', type=click.Path(exists=True, dir_okay=False))
def main(predictions):
    df = pd.read_hdf(predictions)
    plt.hist(df.predictions_convnet, bins=np.linspace(0, 1, 50))
    plt.show()


if __name__ == '__main__':
    main()
