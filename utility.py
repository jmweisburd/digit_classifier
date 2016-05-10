import pickle
import numpy as np
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from skimage.feature import hog
from skimage import data, color, exposure

def load_training_data():
    data_path = "pickle/training.pickle"
    if (os.path.isfile(data_path)):
        with open(data_path, 'rb') as handle:
            td = pickle.load(handle)
        return td
    else:
        td = np.genfromtxt("data/train.csv", delimiter=',', skip_header = 1)
        with open(data_path, 'wb') as handle:
            pickle.dump(td, handle)
        return td

def process_hog_features(raw_training_data):
    data_path = "pickle/hog.pickle"
    if (os.path.isfile(data_path)):
        with open(data_path, 'rb') as handle:
            hf = pickle.load(handle)
        return hf
    else:
        hf = calculate_hog_features(raw_training_data)
        with open(data_path, 'wb') as handle:
            pickle.dump(hf, handle)
        return hf

def calculate_hog_features(raw_training_data):
    num_train = len(raw_training_data)
    test = raw_training_data[0]
    test = test.reshape((28,28))
    fd = hog(test, orientations=12, pixels_per_cell=(7,7), cells_per_block=(1,1), visualise=False)
    hf_length = len(fd)
    hog_features = np.zeros((num_train,hf_length))
    for i in range(num_train):
        temp = raw_training_data[i]
        temp = temp.reshape((28,28))
        fd = hog(temp, orientations=12, pixels_per_cell=(7,7), cells_per_block=(1,1), visualise=False)
        hog_features[i] = fd

    return hog_features

def split_data(feature_vectors, raw_training_label, m):
    num_total = feature_vectors.shape[0]
    num_features = feature_vectors.shape[1]
    training_data = np.zeros(((num_total-(num_total*.1)), num_features))
    testing_data = np.zeros(((num_total*.1), num_features))
    training_label = []
    testing_label = []
    training_count = 0
    testing_count = 0
    for i in range(num_total):
        if i % 10 == m:
            testing_data[testing_count] = feature_vectors[i]
            testing_label.append(raw_training_label[i])
            testing_count += 1
        else:
            training_data[training_count] = feature_vectors[i]
            training_label.append(raw_training_label[i])
            training_count += 1

    return training_data, training_label, testing_data, testing_label
