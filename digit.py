import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

from skimage.feature import hog
from skimage import data, color, exposure
from utility import *
from sklearn import svm
from sklearn.metrics import accuracy_score

raw_training_data = load_training_data()
raw_training_label = np.copy(raw_training_data[:,0])
raw_training_data = np.delete(raw_training_data, 0, 1)

hog_features = process_hog_features(raw_training_data)

print(hog_features.shape)

training_data, training_label, testing_data, testing_label = split_data(hog_features, raw_training_label,1)

clf = svm.LinearSVC()
clf.fit(training_data, training_label)
predicited = clf.predict(testing_data)

print(accuracy_score(testing_label, predicited))
