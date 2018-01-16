import numpy as np
#from functools import partial
import photon_stream as ps
from tqdm import tqdm
import h5py
from fact.io import initialize_h5py, append_to_h5py
from astropy.table import Table
from glob import glob
import click
#from fact.io import read_h5py
import os
import pickle
from joblib import Parallel, delayed
from itertools import islice
from numba import jit


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


mapping = pickle.load(open('./01_hexagonal_position_dict.p', 'rb'))
mapping = np.array([t[1] for t in mapping.items()])


@jit
def make_image(pixel_mapping, photon_content):
    input_matrix = np.zeros([46, 45])
    for i in range(1440):
        x, y = pixel_mapping[i]
        input_matrix[int(x)][int(y)] = photon_content[i]
    return input_matrix


def convert_event(event, roi=[5, 40]):
    imgs = event.photon_stream.image_sequence
    m = imgs[roi[0]:roi[1]].sum(axis=0)
    return make_image(mapping, m)


def image_from_event(event):
    night = int(event.observation_info.night)
    run = int(event.observation_info.run)
    event_num = int(event.observation_info.event)
    img = convert_event(event)
    return [night, run, event_num, event.az, event.zd], img


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
@click.argument('out_file', type=click.Path(exists=False, dir_okay=False))
def main(out_file):
    '''
    Reads all photon_stream files in data/photonstrem/ and converts them to images including
    '''

    if os.path.exists(out_file):
        click.confirm('Do you want to overwrite existing file?', abort=True)
        os.remove(out_file)


    files = sorted(list(glob('./data/photonstream/*.phs.jsonl.gz')))
    file_chunks = np.array_split(files, 9)
    for fs in tqdm(file_chunks):
        results = Parallel(n_jobs=24, verbose=36,  backend='multiprocessing')(map(delayed(convert_file), fs))

        for r in results:
            try:
                write_to_hdf(r, out_file)
            except:
                pass
                #print('Not writing result {} to file'.format(r))

        print('Successfully read {} files'.format(len(results)))


if __name__ == '__main__':
    main()
