import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

fashion = keras.datasets.fashion_mnist
(xtrain, ytrain), (xtest,ytest) = fashion.load_data()