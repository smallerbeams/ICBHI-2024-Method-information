import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tsaug as tsa

from numba import cuda
from natsort import natsorted
from sklearn.utils import shuffle
from scipy.io import loadmat,savemat
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


from model_factory_Gen import AE_CNN_246_shaopu_fristlayer_plus_Dense64_class,\
                AE_CNN_246_shaopu_fristlayer_plus_Dense128_level,AE_CNN_246_shaopu_thirdlayer_plus_Dense128_level


import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, BatchNormalization

def normalize(input_list):

    x_nor = np.copy(input_list)

    for n,signal in enumerate(input_list):

        x_max = np.max(signal)
        x_min = np.min(signal)
        x_mean = np.mean(signal)

        x_nor[n] = (signal-x_min)/(x_max-x_min)

    return x_nor

def data_gen_246_reslut(data_ls, data_pre_fix):
    BatchData_mri = list()
    case_name_list = list()
    time_name_list = list()
    data_ls = natsorted(data_ls)
    for img_p in data_ls:

        mat_data = loadmat(data_pre_fix + img_p)
        mri = mat_data['mri_d']

        mri_n = normalize(np.array(mri))
        mri_n = np.transpose(mri_n, axes=[1,0])
        sort_BatchData_mri=list()
        sort_BatchData_vid=list()
        case_name_list.append(img_p.split('.')[0].split('_')[0])
        time_name_list.append(int(img_p.split('.')[0].split('_')[-1])+1)
        BatchData_mri.append(mri_n.astype(np.float32))

    return np.array(BatchData_mri),case_name_list,time_name_list


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    csv_split = 'D:/ICBHI2024/Dataset/Data/Supplementary/submission.csv'##offical submit format
    data_root = 'D:/ICBHI2024/Dataset/Test_Data_Mat/'##data root
    save_path = 'D:/ICBHI2024/Result'##save root
    save_csv_name = 'paper_use_select-data_class_pretrain.csv'##save name
    reslut_df = pd.read_csv(csv_split)

    total_data_list = os.listdir(data_root)

    data_total, case_name_total, time_name_total = data_gen_246_reslut(total_data_list,data_root)

    model = AE_CNN_246_shaopu_fristlayer_plus_Dense64_class()
    model.summary()
    # model.load_weights('D:/ICBHI2024/Code/weight/AE_CNN_246_shaopu_fristlayer_plus_Dense64_class/AE_CNN_246_shaopu_fristlayer_plus_Dense64_class1467.h5')
    # model.load_weights('D:/ICBHI2024/Code/weight/CNN_246_shaopu_fristlayer_plus_Dense64_class/CNN_246_shaopu_fristlayer_plus_Dense64_class702.h5')
    model.load_weights('D:/ICBHI2024/Code/weight/AE_CNN_246_shaopu_phase2_select_fristlayer_plus_Dense64_pretrain_acc_class/AE_CNN_246_shaopu_phase2_select_fristlayer_plus_Dense64_pretrain_acc_class719.h5')
    # model.load_weights('D:/ICBHI2024/Code/weight/AE_CNN_246_shaopu_phase2_select_add-test-acc_fristlayer_plus_Dense64_class/AE_CNN_246_shaopu_phase2_select_add-test-acc_fristlayer_plus_Dense64_class1609.h5')

    pred = model.predict(data_total)
    result_pred = np.argmax(pred,axis=-1).astype(int)

    for num in range(len(case_name_total)):
        reslut_df.loc[num,'CLASS']=(result_pred[num]-1).astype(int)


    model = AE_CNN_246_shaopu_fristlayer_plus_Dense128_level()
    # model = AE_CNN_246_shaopu_thirdlayer_plus_Dense128_level()
    model.summary()
    breakpoint()
    # model.load_weights('D:/ICBHI2024/Code/weight/AE_CNN_246_shaopu_fristlayer_plus_Dense128_level/AE_CNN_246_shaopu_fristlayer_plus_Dense128_level318.h5')
    # model.load_weights('D:/ICBHI2024/Code/weight/CNN_246_shaopu_fristlayer_plus_Dense128_level/CNN_246_shaopu_fristlayer_plus_Dense128_level414.h5')
    model.load_weights('D:/ICBHI2024/Code/weight/AE_CNN_246_shaopu_phase2_select_fristlayer_plus_Dense128_level/AE_CNN_246_shaopu_phase2_select_fristlayer_plus_Dense128_level2386.h5')
    # model.load_weights('D:/ICBHI2024/Code/weight/AE_CNN_246_shaopu_phase2_select_add-test-acc_fristlayer_plus_Dense128_level/AE_CNN_246_shaopu_phase2_select_add-test-acc_fristlayer_plus_Dense128_level1462.h5')
    # model.load_weights('D:/ICBHI2024/Code/weight/AE_CNN_246_shaopu_thirdlayer_plus_Dense128_level/AE_CNN_246_shaopu_thirdlayer_plus_Dense128_level117.h5')
    pred = model.predict(data_total)
    result_pred = np.argmax(pred,axis=-1).astype(int)

    for num in range(len(case_name_total)):
        reslut_df.loc[num,'LEVEL']=(result_pred[num]-4).astype(int)


    print(reslut_df)
    reslut_df.to_csv(save_path+'/'+save_csv_name,index = False)


