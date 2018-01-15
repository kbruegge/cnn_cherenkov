import click
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fact.analysis import li_ma_significance
import fact.io as fio

stats_box_template = r'''
$N_\mathrm{{On}} = {n_on}$, $N_\mathrm{{Off}} = {n_off}$, $\alpha = {alpha}$
$N_\mathrm{{Exc}} = {n_excess:.1f} \pm {n_excess_err:.1f}$, $S_\mathrm{{Li&Ma}} = {significance:.1f}\,\sigma$
'''


@click.command()
@click.argument('predictions', type=click.Path(exists=True, dir_okay=False))
@click.option('-t', '--threshold', default=0.5)
@click.option('-c', '--theta_cut', default=0.01)
@click.option('--net/--no-net', default=True)
def main(predictions, threshold, theta_cut, net):
    bins = 40
    alpha = 0.2
    limits = [0, 0.3]
    df = fio.read_data(predictions, key='events')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if net:
        print('using cnn predictions')
        selected = df.query('predictions_convnet > {}'.format(threshold))
        ax.set_title('Neural Net predictions')
    else:
        print('using standard predictions')
        selected = df.query('gamma_prediction > {}'.format(threshold))
        ax.set_title('RF predictions')

    theta_on = selected.theta_deg
    theta_off = pd.concat([
        selected['theta_deg_off_{}'.format(i)]
        for i in range(1, 6)
    ])
    h_on, bin_edges = np.histogram(
        theta_on.apply(lambda x: x**2).values,
        bins=bins,
        range=limits
    )
    h_off, bin_edges, _ = ax.hist(
        theta_off.apply(lambda x: x**2).values,
        bins=bin_edges,
        range=limits,
        weights=np.full(len(theta_off), 0.2),
        histtype='stepfilled',
        color='lightgray',
    )

    bin_center = bin_edges[1:] - np.diff(bin_edges) * 0.5
    bin_width = np.diff(bin_edges)

    ax.errorbar(
        bin_center,
        h_on,
        yerr=np.sqrt(h_on) / 2,
        xerr=bin_width / 2,
        linestyle='',
        label='On',
    )
    ax.errorbar(
        bin_center,
        h_off,
        yerr=alpha * np.sqrt(h_off) / 2,
        xerr=bin_width / 2,
        linestyle='',
        label='Off',
        color = 'darkgray',
    )

    ax.axvline(theta_cut**2, color='black', alpha=0.3, linestyle='--')

    n_on = np.sum(theta_on < theta_cut)
    n_off = np.sum(theta_off < theta_cut)
    significance = li_ma_significance(n_on, n_off, alpha=alpha)

    print('N_on', n_on)
    print('N_off', n_off)
    print('Li&Ma: {}'.format(significance))

    ax.text(
        0.5, 0.95,
        stats_box_template.format(
            n_on=n_on, n_off=n_off, alpha=alpha,
            n_excess=n_on - alpha * n_off,
            n_excess_err=np.sqrt(n_on + alpha**2 * n_off),
            significance=significance,
        ),
        transform=ax.transAxes,
        va='top',
        ha='center',
    )


    ax.set_xlim(*limits)
    ax.legend(loc='lower right')
    fig.tight_layout(pad=0)

    plt.show()



if __name__ == '__main__':
    main()
