
import numpy as np

import h5py
import code


def real_and_imag_channel(file, mask):
    f = h5py.File(file, 'r')
    CH_RE = np.array(f[mask])['real']
    CH_RE = np.transpose(CH_RE, [2, 1, 0])
    CH_IMAG = np.array(f[mask])['imag']
    CH_IMAG = np.transpose(CH_IMAG, [2, 1, 0])
    return CH_RE.astype('float32'), CH_IMAG.astype('float32')


def train_test_split(X):
    begin_test = int(np.floor(X.shape[0] * 0.9))
    end_test = X.shape[0]
    X_test = X[begin_test:end_test]
    begin_val = int(np.floor(X.shape[0] * 0.8))
    X_train = X[0:begin_val]
    X_val = X[begin_val:begin_test]
    return X_train, X_val, X_test




def get_dataset(direction='uplink', scenario_name='berlin', gap='2', dim_c=1):

    if direction == 'uplink':
        filename = scenario_name + '/ch_ul.mat'
        mask_n = 'ch_ul'

    elif direction == 'downlink':
        filename = scenario_name + '/ch_dl_' + gap + '.mat'
        mask_n = 'ch_dl_' + gap

    data_real, data_imag = real_and_imag_channel(file=filename, mask=mask_n)

    x_train_re, x_val_re, x_test_re = train_test_split(data_real)
    x_train_im, x_val_im, x_test_im = train_test_split(data_imag)


    x_train_re = np.expand_dims(x_train_re, axis=dim_c)
    x_train_im = np.expand_dims(x_train_im, axis=dim_c)

    x_val_re = np.expand_dims(x_val_re, axis=dim_c)
    x_val_im = np.expand_dims(x_val_im, axis=dim_c)

    x_test_re = np.expand_dims(x_test_re, axis=dim_c)
    x_test_im = np.expand_dims(x_test_im, axis=dim_c)

    x_train = np.concatenate((x_train_re, x_train_im), axis=dim_c)
    x_val = np.concatenate((x_val_re, x_val_im), axis=dim_c)
    x_test = np.concatenate((x_test_re, x_test_im), axis=dim_c)

    return x_train, x_val, x_test






def get_dl_test_set(scenario_name='berlin', gap='2'):

    if gap == '1':
        filename = scenario_name + '/ch_dl.mat'
        mask_n = 'ch_dl'
    else:
        filename = scenario_name + '/ch_dl_' + gap + '.mat'
        mask_n = 'ch_dl_' + gap

    data_real, data_imag = real_and_imag_channel(file=filename, mask=mask_n)

    x_train_re, x_val_re, x_test_re = train_test_split(data_real)
    x_train_im, x_val_im, x_test_im = train_test_split(data_imag)

    x_test_re = np.expand_dims(x_test_re, axis=1)
    x_test_im = np.expand_dims(x_test_im, axis=1)

    x_test = np.concatenate((x_test_re, x_test_im), axis=1)


    return x_test

