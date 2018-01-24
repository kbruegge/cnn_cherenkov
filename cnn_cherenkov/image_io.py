import h5py
import pandas as pd
import fact.io as fio
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle


def load_mc_training_data(N=-1):
    N = N // 2
    df_gammas, images_gammas = read_rows('./data/gamma_images.hdf5', N=N)
    df_protons, images_protons = read_rows('./data/proton_images.hdf5', N=N)

    images_gammas = scale_images(images_gammas)
    images_protons = scale_images(images_protons)
    X = np.vstack([images_gammas, images_protons])

    Y = np.append(np.ones(len(df_gammas)), np.zeros(len(df_protons)))
    Y = OneHotEncoder().fit_transform(Y.reshape(-1, 1)).toarray()

    #X, Y = shuffle(X, Y)
    print('wow not teh shuffling')
    return X, Y


def scale_images(images):
    images[images < 3] = 0
    qmax = np.percentile(images, q=99.5, axis=(1, 2))
    a = images / qmax[:, np.newaxis, np.newaxis]
    #a = images[:, :, :, np.newaxis]
    return a.reshape((len(images), 46, 45, -1))


def read_rows(path, N=-1):
    '''
    read given rows from hdf5 images.
    return dataframe containg high level information and images (df, images)
    '''
    f = h5py.File(path)
    if N > 1:
        night = f['events/night'][0:N]
        run = f['events/run_id'][0:N]
        event = f['events/event_num'][0:N]
        az = f['events/az'][0:N]
        zd = f['events/zd'][0:N]
        images = f['events/image'][0:N]
    else:
        night = f['events/night'][:]
        run = f['events/run_id'][:]
        event = f['events/event_num'][:]
        az = f['events/az'][:]
        zd = f['events/zd'][:]
        images = f['events/image'][:]


    df = pd.DataFrame({'night': night, 'run_id': run, 'event_num': event, 'zd': zd, 'az': az, })
    return df, images


def load_crab_training_data(N=-1, prediction_threshold=0.8):
    '''
    Returns array of images X and one-hot encoded labels Y. Both classes equally sampled.
    '''

    dl3 = fio.read_data('./data/dl3/open_crab_sample_dl3.hdf5', key='events')
    dl3 = dl3.set_index(['night', 'run_id', 'event_num'])

    print('Reading crab image index')
    f = h5py.File('./data/crab_images.hdf5')
    night = f['events/night'][:]
    run = f['events/run_id'][:]
    event = f['events/event_num'][:]

    df = pd.DataFrame({'night': night, 'run_id': run, 'event_num': event})
    df['int_index'] = df.index
    df = df.set_index(['night', 'run_id', 'event_num'])

    print('joining crab data with analysis results')
    data = df.join(dl3, how='inner')

    indices = data.int_index.values
    data = data.set_index(indices)

    if N > 0:
        indices = indices[:N]

    print('sorting index')
    indices = list(sorted(indices))

    print('loading {} images'.format(len(indices)))
    images = f['events/image'][indices]

    print('selecting types')
    data = data.loc[indices]
    data = data.reset_index()

    gammas = data[data.gamma_prediction >= 0.8]
    protons = data[data.gamma_prediction < 0.8]

    ids_gamma = np.random.choice(gammas.index.values, N // 2)
    ids_proton = np.random.choice(protons.index.values, N // 2)
    ids = np.append(ids_gamma, ids_proton)

    X = images[ids]
    Y = np.where(data.loc[ids].gamma_prediction > 0.8, 1.0, 0.0)
    df = data.copy().loc[ids]

    print('Loaded {} positive labels and {} negative labels'.format(np.sum(Y), N - np.sum(Y)))

    Y = OneHotEncoder().fit_transform(Y.reshape(-1, 1)).toarray()

    df, X, Y = shuffle(df, X, Y)
    X = scale_images(X)

    return df, X, Y



def number_of_images(path):
    '''
    return number of images in hdf5 file
    '''
    f = h5py.File(path)
    return len(f['events/night'])
