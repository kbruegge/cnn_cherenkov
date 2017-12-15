import numpy as np
import photon_stream as ps
from tqdm import tqdm
import h5py
from fact.io import initialize_h5py, append_to_h5py
from astropy.table import Table
from glob import glob
import click
from fact.io import read_h5py
import os
import pickle


mapping = pickle.load(open('./data/01_hexagonal_position_dict.p', 'rb'),)


def make_image(mapping, photon_content):
    input_matrix = np.zeros([46, 45])
    for i in range(1440):
        x, y = mapping[i]
        input_matrix[int(x)][int(y)] = photon_content[i]
    return input_matrix

def convert_event(event, roi=[5, 40], threshold=3):
    imgs = event.photon_stream.image_sequence
    m = imgs[roi[0]:roi[1]].sum(axis=0)
    m = np.clip(m, threshold, m.max())
    return make_image(mapping, m)


def recarray_from_ps_reader(reader, labeled_events):
    rows = []
    imgs = []

    for event in tqdm(reader):
        try:
            night = int(event.observation_info.night)
            run = int(event.observation_info.run)
            event_num = int(event.observation_info.event)
            p = labeled_events.loc[(night, run, event_num), :]

            rows.append([night, run, event_num, event.az, event.zd, p.gamma_prediction, p.gamma_energy_prediction, p.ra_prediction, p.dec_prediction])
            img = convert_event(event)
            imgs.append(img)
        except KeyError:
            pass

    rows = np.array(rows)
    columns = [rows[:, i] for i in range(9)]

    rows = np.array(rows)
    t = Table(
        [*columns, imgs],
        names=('night', 'run', 'event', 'az', 'zd', 'gamma_prediction', 'gamma_energy_prediction', 'ra_prediction', 'dec_prediction', 'image')
    ).as_array()

    return t


def write_to_hdf(recarray, filename):
    key = 'events'
    with h5py.File(filename, mode='a') as f:
        print('Writing {} events to file'.format(len(recarray)))
        if key not in f:
            initialize_h5py(f, recarray, key=key)
        append_to_h5py(f, recarray, key=key)


@click.command()
@click.argument('dl3_events', type=click.Path(exists=True, dir_okay=False))
@click.argument('out_file', type=click.Path(exists=False, dir_okay=False))
def main(dl3_events, out_file):
    '''
    Rads all photon_stream files in data/photonstrem/ and converts them to images including
    important data from dl3 (pointing and such).

    This runs a while. (like hours). somebody could sure add multiprocessing.
    '''

    if os.path.exists(out_file):
        click.confirm('Do you want to overwrite existing file?', abort=True)
        os.remove(out_file)

    labeled_events = read_h5py(dl3_events, key='events').set_index(['night', 'run_id', 'event_num'])

    files = sorted(list(glob('./data/photonstream/*.phs.jsonl.gz')))
    for photon_stream_file in files:
        print('Analyzing {}'.format(photon_stream_file))
        reader = ps.EventListReader(photon_stream_file)
        recarray = recarray_from_ps_reader(reader, labeled_events)
        write_to_hdf(recarray, out_file)


if __name__ == '__main__':
    main()
