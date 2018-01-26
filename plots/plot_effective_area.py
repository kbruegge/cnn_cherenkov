import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import click
from astropy.stats import binom_conf_interval
import fact.io as fio


def histograms(
        all_events,
        selected_events,
        bins,
        range=None,
        log=True,
        ):
    '''
    Create histograms in the given bins for two vectors.
    Parameters
    ----------
    all_events: array-like
        Quantity which should be histogrammed for all simulated events
    selected_events: array-like
        Quantity which should be histogrammed for all selected events
    bins: int or array-like
        either number of bins or bin edges for the histogram
    range: (float, float)
        The lower and upper range of the bins
    log: bool
        flag indicating whether log10 should be applied to the values.
    returns: hist_all, hist_selected,  bin_edges
    '''

    if log is True:
        all_events = np.log10(all_events)
        selected_events = np.log10(selected_events)

    hist_all, bin_edges = np.histogram(
        all_events,
        bins=bins,
        range=range,
    )

    hist_selected, _ = np.histogram(
        selected_events,
        bins=bin_edges,
    )

    return hist_all, hist_selected, bin_edges



@u.quantity_input(impact=u.meter)
def collection_area(
        all_events,
        selected_events,
        impact,
        bins,
        range=None,
        log=True,
        sample_fraction=1.0,
        ):
    '''
    Calculate the collection area for the given events.
    Parameters
    ----------
    all_events: array-like
        Quantity which should be histogrammed for all simulated events
    selected_events: array-like
        Quantity which should be histogrammed for all selected events
    bins: int or array-like
        either number of bins or bin edges for the histogram
    impact: astropy Quantity of type length
        The maximal simulated impact parameter
    log: bool
        flag indicating whether log10 should be applied to the quantity.
    sample_fraction: float
        The fraction of `all_events` that was analysed
        to create `selected_events`
    '''

    hist_all, hist_selected, bin_edges = histograms(
        all_events,
        selected_events,
        bins,
        range=range,
        log=log
    )

    hist_selected = (hist_selected / sample_fraction).astype(int)

    bin_width = np.diff(bin_edges)
    bin_center = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    invalid = hist_selected > hist_all
    hist_selected[invalid] = hist_all[invalid]
    # use astropy to compute errors on that stuff
    lower_conf, upper_conf = binom_conf_interval(hist_selected, hist_all)

    # scale confidences to match and split
    lower_conf = lower_conf * np.pi * impact**2
    upper_conf = upper_conf * np.pi * impact**2

    area = hist_selected / hist_all * np.pi * impact**2

    return area, bin_center, bin_width, lower_conf, upper_conf



@click.command()
@click.argument('predictions', type=click.Path(exists=True, dir_okay=False))
@click.option('-t', '--threshold', default=0.5)
@click.option('-b', '--bins', default=10)
def main(predictions, threshold, bins):
    df = fio.read_data(predictions, key='events')
    df = df[df.predictions_convnet >= threshold]
    selected_event_energies = df.energy.values

    df = fio.read_data(predictions, key='showers')
    all_event_energies = df.energy.values
    ax = plt.gca()

    bins = np.logspace(
        np.log10(all_event_energies.min()),
        np.log10(all_event_energies.max()),
        bins + 1,
    )

    ret = collection_area(
        all_event_energies,
        selected_event_energies,
        impact=270 * u.m,
        bins=bins,
        log=False,
        sample_fraction=1.0,
    )
    area, bin_centers, bin_width, lower_conf, upper_conf = ret

    ax.errorbar(
        bin_centers,
        area.value,
        xerr=bin_width / 2,
        yerr=[
            (area - lower_conf).value,
            (upper_conf - area).value
        ],
    )

    plt.yscale('log')
    plt.xscale('log')

    plt.show()


if __name__ == '__main__':
    main()
