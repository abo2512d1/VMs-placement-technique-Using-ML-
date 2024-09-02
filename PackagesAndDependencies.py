# import libraries
import tensorflow as tf
import keras
import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
# from __future__ import division, print_function, absolute_import
#from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestCentroid
import seaborn as sns
from sklearn.metrics import classification_report
import pylab as pl
import pandas as pd
import shutil
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import silhouette_score
from statistics import stdev
import copy
from keras.models import Model
import tensorflow as tf
from tensorflow import keras
from keras import Input
from tensorflow.keras import layers
from keras.layers import Dense
from keras.models import Model
fig = plt.figure(figsize=(20,10))
import random