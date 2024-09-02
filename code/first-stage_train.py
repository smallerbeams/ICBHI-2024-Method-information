import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tsaug as tsa

from sklearn.utils import shuffle
from scipy.io import loadmat,savemat
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


from model_factory import AE_CNN_246_shaopu


import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, BatchNormalization


def one_hot(lab, num_class):
    return np.eye(num_class)[lab]

def data_read(csv_path):

    df = pd.read_csv(csv_path)

    train_list = df.iloc[:,0].tolist()
    valid_list = df.iloc[:,1].tolist()
    test_list = df.iloc[:,2].tolist()

    train_list = [x.split('\\')[-1].split('_')[0] + '_' + x.split('\\')[-1].split('_')[1] + '.mat' for x in train_list]
    valid_list = [x.split('\\')[-1].split('_')[0] + '_' + x.split('\\')[-1].split('_')[1] + '.mat' for x in valid_list if x != "0"]
    test_list = [x.split('\\')[-1].split('_')[0] + '_' + x.split('\\')[-1].split('_')[1] + '.mat' for x in test_list if x != "0"]

    return train_list, valid_list, test_list

def data_gen_fn(data_ls, data_pre_fix, batch_size=16,aug=True):
    data_path_ls = shuffle(data_ls)

    i = 0

    augmenter = (
        tsa.AddNoise(scale=0.05) @ 0.5 +
        tsa.TimeWarp(n_speed_change=5, max_speed_ratio=5) @ 0.5 +
        tsa.Drift(max_drift=0.4, n_drift_points=3) @ 0.5
        )

    while True:
        start = i * batch_size
        end = np.min([start + batch_size, len(data_path_ls)])
        BatchData_mri = list()
        BatchData_ppg = list()
        BatchData_rep = list()

        BatchData_p = list()

        for img_p in data_path_ls[start:end]:
            filter_fmri = list()
            BatchData_p.append(img_p)
            
            mat_data = loadmat(data_pre_fix + img_p)
            mri = mat_data['mri_d']

            mri_n = normalize(np.array(mri))

            mri_n = np.transpose(mri_n, axes=[1,0])
            if aug:
                mri_n = augmenter.augment(mri_n)

            BatchData_mri.append(mri_n.astype(np.float32))

        yield np.array(BatchData_mri), \
              np.array(BatchData_mri).astype(np.float32),


        i = i+1
        if (i * batch_size) >= (len(data_ls)):
            data_path_ls = shuffle(data_ls)
            i = 0

def normalize(input_list):

    x_nor = np.copy(input_list)

    for n,signal in enumerate(input_list):

        x_max = np.max(signal)
        x_min = np.min(signal)
        x_mean = np.mean(signal)

        x_nor[n] = (signal-x_min)/(x_max-x_min)

    return x_nor

######## train loading
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    data_root = 'E:/ICBHI2024/Dataset/AE_Data_Mat/'
    data_root_test = 'E:/ICBHI2024/Dataset/Test_Data_Mat/'
    save_weight = 'E:/ICBHI2024/Code/weight/AE_CNN_246_shaopu'
    weight_name = 'AE_CNN_246_shaopu'

    total_data_list = os.listdir(data_root)
    test_data_name = os.listdir(data_root_test)

    x_train, x_val, x_test = data_read(csv_split)

    epochs = 3000
    batch_size = 32
    train_data_gn = data_gen_fn(total_data_list, data_root, batch_size=batch_size,aug=False)
    val_data_gn = data_gen_fn(test_data_name, data_root, batch_size=1,aug=False)

    steps_train_epoch = int(np.ceil(len(total_data_list) / batch_size))
    steps_val_epoch = int(np.ceil(len(test_data_name) / 1))

    model = AE_CNN_246_shine()
    model.summary()
    breakpoint()

    mse = tf.keras.losses.MeanSquaredError()
    adam = tf.optimizers.Adam(learning_rate=1e-4)

    mse_score = tf.keras.metrics.MeanSquaredError()
    mae_score = tf.keras.metrics.MeanAbsoluteError(name='mae')
    model.compile(optimizer=adam, loss=mse, metrics=[mae_score])

    ##### callbacks model monitor
    model_path = save_weight + '/'+ weight_name + '{epoch:03d}.h5'
    if not os.path.isdir(save_weight):
        os.makedirs(save_weight)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_mae', save_best_only=True, verbose=1, mode='min')
    csv_logger = tf.keras.callbacks.CSVLogger(save_weight+'/'+ weight_name+'_log.csv')
    sch = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.1,
                patience=5,
                verbose=0,
                mode='auto',
                min_delta=0.01,
                cooldown=0,
                min_lr=1e-5)

    ##### model training
    history = model.fit(train_data_gn,
            epochs=epochs, verbose='auto',
            steps_per_epoch=steps_train_epoch,
            validation_data=val_data_gn,
            validation_steps=steps_val_epoch,
            callbacks=[csv_logger])
    model.save_weights(save_weight+'/'+ weight_name+'_final_weight.h5')
