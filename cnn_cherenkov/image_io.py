import h5py
import pandas as pd
import fact.io as fio
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from tqdm import tqdm


def load_mc_training_data(N=-1, path_gamma='./data/gamma_images.hdf5', path_proton='./data/proton_images.hdf5'):
    '''
    Loads gamma and proton images from hdf5 files and returns images X and One-Hot encoed labels Y
    '''
    N = N // 2
    df_gammas, images_gammas = read_rows(path_gamma, N=N)
    df_protons, images_protons = read_rows(path_proton, N=N)

    images_gammas = scale_images(images_gammas)
    images_protons = scale_images(images_protons)
    X = np.vstack([images_gammas, images_protons])

    Y = np.append(np.ones(len(df_gammas)), np.zeros(len(df_protons)))
    Y = OneHotEncoder().fit_transform(Y.reshape(-1, 1)).toarray()

    X, Y = shuffle(X, Y)
    return X, Y


def load_mc_data_mix(N=-1, prediction_threshold=0.8):
    '''
    Loads MC data mixed with real observation data for trained. Results are shuffled.
    '''
    _, X_data, Y_data = load_crab_training_data(N=N // 2, prediction_threshold=prediction_threshold)
    X_mc, Y_mc = load_mc_training_data(N=N // 2)

    X = np.vstack([X_data, X_mc])
    Y = np.vstack([Y_data, Y_mc])

    X, Y = shuffle(X, Y, random_state=0)
    return X, Y


def scale_images(images):
    qmax = np.percentile(images, q=99.5, axis=(1, 2))
    a = images / qmax[:, np.newaxis, np.newaxis]
    return a.reshape((len(images), 45, 46, -1))


def read_rows(path, N=-1):
    '''
    read given rows from hdf5 images.
    return dataframe containg high level information and images (df, images)
    '''
    f = h5py.File(path)
    if N < 1:
        N = len(f['events/image'])

    d = {}
    if 'corsika_phi' in list(f['events']):
        d['reuse'] = f['events/reuse'][0:N]
        d['run'] = f['events/run'][0:N]
        d['event'] = f['events/event_num'][0:N]
        d['energy'] = f['events/energy'][0:N]
        d['impact_x'] = f['events/impact_x'][0:N]
        d['impact_y'] = f['events/impact_y'][0:N]
        d['corsika_phi'] = f['events/corsika_phi'][0:N]
        d['corsika_phi'] = f['events/corsika_theta'][0:N]

    else:
        d['night'] = f['events/night'][0:N]
        d['run'] = f['events/run_id'][0:N]
        d['event'] = f['events/event_num'][0:N]
        d['az'] = f['events/az'][0:N]
        d['zd'] = f['events/zd'][0:N]

    images = f['events/image'][0:N]

    return pd.DataFrame(d), images


def load_crab_training_data(N=-1, prediction_threshold=0.8):
    '''
    Returns array of images X and one-hot encoded labels Y. Both classes equally sampled.
    '''

    dl3 = fio.read_data('./data/dl3/open_crab_sample_dl3.hdf5', key='events')
    dl3 = dl3.set_index(['night', 'run_id', 'event_num'])

    f = h5py.File('./data/crab_images.hdf5', 'r')
    night = f['events/night'][:]
    run = f['events/run_id'][:]
    event = f['events/event_num'][:]

    df = pd.DataFrame({'night': night, 'run_id': run, 'event_num': event})
    df['int_index'] = df.index
    df = df.set_index(['night', 'run_id', 'event_num'])

    data = df.join(dl3, how='inner')

    indices = data.int_index.values
    data = data.set_index(indices)

    indices = list(sorted(indices))

    if N > 0:
        indices = indices[:N]
    else:
        N = len(indices)


    print('loading {} images'.format(len(indices)))
    images = load_images_with_index(indices)

    data = data.loc[indices]
    data = data.reset_index()

    gammas = data[data.gamma_prediction >= prediction_threshold]
    protons = data[data.gamma_prediction < prediction_threshold]

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


def load_images_with_index(indices, path='./data/crab_images.hdf5'):
    '''
    Gets images from file given by path. Reads all images in batches of 10000.
    All images in one batch are read and then selected using the index. This is about two
    orders of magnitude faster then using the indices to select from the h5py object directly
    for some reason.
    '''
    f = h5py.File(path)

    # create selections
    N = indices[-1]
    idx = np.arange(0, N + 1)
    mask = np.zeros_like(idx, dtype=np.bool)
    mask[indices] = True

    # split into batches
    number_of_sections = max(N / 10000, 1)
    idx = np.array_split(np.arange(0, N + 1), number_of_sections)
    masks = np.array_split(mask, number_of_sections)

    images = []
    for ids, selection in tqdm(zip(idx, masks)):
        l = ids[0]
        u = ids[-1]
        selected_images = f['events/image'][l:u + 1][selection]
        images.append(selected_images)

    return np.vstack(images)


def load_crab_data(start=0, end=1000,):
    dl3 = fio.read_data('./data/dl3/open_crab_sample_dl3.hdf5', key='events')
    dl3 = dl3.set_index(['night', 'run_id', 'event_num'])

    f = h5py.File('./data/crab_images.hdf5', 'r')
    night = f['events/night'][start:end]
    run = f['events/run_id'][start:end]
    event = f['events/event_num'][start:end]
    images = f['events/image'][start:end]

    df = pd.DataFrame({'night': night, 'run_id': run, 'event_num': event})
    df['int_index'] = df.index
    df = df.set_index(['night', 'run_id', 'event_num'])


    data = df.join(dl3, how='inner')
    images = scale_images(images[data.int_index])

    return data, images



def apply_to_observation_data(model, path='./data/crab_images.hdf5'):
    N = number_of_images(path)
    idx = np.array_split(np.arange(0, N), N / 8000)
    dfs = []
    event_counter = 0
    try:
        for ids in tqdm(idx):
            l = ids[0]
            u = ids[-1]
            df, images = load_crab_data(l, u + 1)
            event_counter += len(df)
            if len(df) == 0:
                continue
            predictions = model.predict(images)[:, 1]

            df['predictions_convnet'] = predictions
            dfs.append(df)
    except KeyboardInterrupt:
        print('User stopped process...')
    except Exception as e:
        print('Aborting due to error:')
        print(e)
    finally:
        print('Concatenating {} data frames'.format(len(dfs)))
        df = pd.concat(dfs)
        assert event_counter == len(df)
        return df



def apply_to_mc(model, path='./data/gamma_images.hdf5', N=-1):
    df_gammas, images = read_rows(path, N=N)
    images = scale_images(images)

    N = len(df_gammas)
    predictions = []
    image_batches = np.array_split(images, N / 8000)
    for batch in tqdm(image_batches):
        p = model.predict(batch)
        predictions.append(p)

    predictions = np.vstack(predictions)[:, 1]
    df_gammas['predictions_convnet'] = predictions
    return df_gammas
