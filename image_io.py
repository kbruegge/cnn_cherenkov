import h5py
import pandas as pd
#import numpy as np


def read_n_rows(path, start=0, end=1000):
    f = h5py.File(path)
    night = f['events/night'][start:end]
    run = f['events/run_id'][start:end]
    event = f['events/event_num'][start:end]
    az = f['events/az'][start:end]
    zd = f['events/zd'][start:end]

    df = pd.DataFrame({'night': night, 'run_id': run, 'event_num': event, 'zd': zd, 'az': az,})
    return df, f['events/image'][start:end]


def number_of_images(path):
    f = h5py.File(path)
    return len(f['events/night'])
