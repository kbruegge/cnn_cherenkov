import h5py
import pandas as pd
import fact.io as fio
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle


def get_mc_training_data(start=0, end=-1):
    df_gammas, images_gammas = read_rows('./data/gamma_images.hdf5', start=start, end=end)
    df_protons, images_protons = read_rows('./data/proton_images.hdf5', start=start, end=end)

    # images_gammas = np.transpose(images_gammas, axes=[0, 2, 1])
    # images_protons = np.transpose(images_protons, axes=[0, 2, 1])

    images_gammas = scale_images(images_gammas)
    images_protons = scale_images(images_protons)
    X = np.vstack([images_gammas, images_protons])

    Y = np.append(np.ones(len(df_gammas)), np.zeros(len(df_protons)))
    Y = OneHotEncoder().fit_transform(Y.reshape(-1, 1)).toarray()

    X, Y = shuffle(X, Y)

    return X, Y


def scale_images(images):
    images[images < 2] = 0
    qmax = np.percentile(images, q=99.8, axis=(1, 2))
    a = images / qmax[:, np.newaxis, np.newaxis]
    #a = images / qmax
    return a.reshape((len(images), 46, 45, -1))


def read_rows(path, start=0, end=1000):
    '''
    read given rows from carb images.
    return dataframe containg high level infor and iimages (df, images)
    '''
    f = h5py.File(path)
    night = f['events/night'][start:end]
    run = f['events/run_id'][start:end]
    event = f['events/event_num'][start:end]
    az = f['events/az'][start:end]
    zd = f['events/zd'][start:end]

    df = pd.DataFrame({'night': night, 'run_id': run, 'event_num': event, 'zd': zd, 'az': az, })
    return df, f['events/image'][start:end]


def load_crab_data(start=0, end=-1):
    '''
    Loads images and crab dl3 data and returns a dataframe and the images.
    '''
    df, images = read_rows('./data/crab_images.hdf5', start=start, end=end)
    dl3 = fio.read_data('./data/dl3/open_crab_sample_dl3.hdf5', key='events')
    dl3 = dl3.set_index(['night', 'run_id', 'event_num'])

    df['int_index'] = df.index
    df = df.set_index(['night', 'run_id', 'event_num'])


    data = df.join(dl3, how='inner')
    print('Events in open data sample: {}, events in photons_stream: {}, events in joined data: {}'.format(len(dl3), len(df), len(data)))

    if len(data) == 0:
        return [], []

    images = scale_images(images[data.int_index])

    assert len(images) == len(data)

    return data, images


def create_training_sample(df, images, prediction_threshold=0.8):
    '''
    Returns array of images X and one-hot encoded labels Y. Both classes equally sampled.
    '''
    df = df.reset_index()
    df['prediction_label'] = np.where(df.gamma_prediction > prediction_threshold, 0, 1)

    Y = df.prediction_label.values.astype(np.float32)

    N = len(df)
    ids_true = df[df.prediction_label == 1].index.values
    ids_true = np.random.choice(ids_true, N // 2)
    ids_false = df[df.prediction_label == 0].index.values
    ids_false = np.random.choice(ids_false, N // 2)
    ids = np.append(ids_false, ids_true)

    X = images[ids]
    Y = Y[ids]
    df = df.copy().loc[ids]

    print('Loaded {} positive labels and {} negative labels'.format(
        np.sum(Y), N - np.sum(Y)))
    Y = OneHotEncoder().fit_transform(Y.reshape(-1, 1)).toarray()
    return df, X, Y




def number_of_images(path):
    '''
    return number of images in hdf5 file
    '''
    f = h5py.File(path)
    return len(f['events/night'])
