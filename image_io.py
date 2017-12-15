import h5py
import pandas as pd
import numpy as np


def read_n_rows(path, start=0, end=1000):
    f = h5py.File(path)
    night = f['events/night'][start:end]
    run = f['events/run'][start:end]
    event = f['events/event'][start:end]
    az = f['events/az'][start:end]
    zd = f['events/zd'][start:end]
    gamma_prediction = f['events/gamma_prediction'][start:end]
    ra_prediction = f['events/ra_prediction'][start:end]
    dec_prediction = f['events/dec_prediction'][start:end]

    df = pd.DataFrame({'night': night, 'run': run, 'event': event, 'zd': zd, 'az': az, 'ra_prediction': ra_prediction, 'dec_prediction': dec_prediction, 'gamma_prediction': gamma_prediction})
    return df, f['events/image'][start:end]


def number_of_images(path):
    f = h5py.File(path)
    return len(f['events/gamma_prediction'])
