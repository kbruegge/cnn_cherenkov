import numpy as np
import photon_stream as ps
from tqdm import tqdm
import h5py
from fact.io import initialize_h5py, append_to_h5py
from astropy.table import Table
import click
import os
import pickle
from joblib import Parallel, delayed
from itertools import islice
from numba import jit


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


mapping = pickle.load(open('./cnn_cherenkov/hexagonal_position_dict.p', 'rb'))
mapping_fact_tools = np.array([t[1] for t in mapping.items()])

@jit
def make_image(photon_content):
    input_matrix = np.zeros([46, 45])
    for i in range(1440):
        x, y = mapping_fact_tools[i]
        input_matrix[int(x)][int(y)] = photon_content[i]
    return input_matrix


def convert_event(event, roi=[5, 40]):
    imgs = event.photon_stream.image_sequence
    m = imgs[roi[0]:roi[1]].sum(axis=0)
    return make_image(m)


def image_from_event(event):
    try:
        night_reuse = int(event.observation_info.night)
        run = int(event.observation_info.run)
        event_num = int(event.observation_info.event)

    except AttributeError:
        night_reuse = int(event.simulation_truth.reuse)
        run = int(event.simulation_truth.run)
        event_num = int(event.simulation_truth.event)

    img = convert_event(event,)
    return [night_reuse, run, event_num, event.az, event.zd], img


def recarray_from_ps_reader(reader):
    rows = []
    imgs = []

    result = list(map(image_from_event, reader))

    rows = np.array([r[0] for r in result])
    imgs = np.array([r[1] for r in result])
    columns = [rows[:, i] for i in range(5)]

    t = Table(
        [*columns, imgs],
        names=('night', 'run_id', 'event_num', 'az', 'zd', 'image')
    ).as_array()

    return t


def write_to_hdf(recarray, filename):
    key = 'events'
    with h5py.File(filename, mode='a') as f:
        if key not in f:
            initialize_h5py(f, recarray, key=key)
        append_to_h5py(f, recarray, key=key)


def convert_file(path):
    # print('Analyzing {}'.format(photon_stream_file))
    reader = ps.EventListReader(path)
    try:
        recarray = recarray_from_ps_reader(reader)
        return recarray
    except ValueError as e:
        print('Failed to read data from file: {}'.format(path))
        print('PhotonStreamError: {}'.format(e))



@click.command()
@click.argument('in_files', nargs=-1)
@click.argument('out_file', type=click.Path(exists=False, dir_okay=False))
@click.option('--n_jobs', '-n', default=4)
@click.option('--n_chunks', '-c', default=9)
def main(in_files, out_file, n_jobs, n_chunks):
    '''
    Reads all photon_stream files and converts them to images.
    '''

    if os.path.exists(out_file):
        click.confirm('Do you want to overwrite existing file?', abort=True)
        os.remove(out_file)


    files = sorted(in_files)
    file_chunks = np.array_split(files, n_chunks)
    for fs in tqdm(file_chunks):
        if n_jobs != 0:
            results = Parallel(n_jobs=n_jobs, verbose=36, backend='multiprocessing')(map(delayed(convert_file), fs))
        else:
            results = map(convert_file, fs)

        for r in results:
            try:
                write_to_hdf(r, out_file)
            except:
                pass

        print('Successfully read {} files'.format(len(results)))


if __name__ == '__main__':
    main()
