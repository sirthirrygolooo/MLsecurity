#Dataset downloading
import kagglehub
import shutil
import os

path = kagglehub.dataset_download("proutkarshtiwari/adni-images-for-alzheimer-detection")
print("Dataset successfully downloaded")
shutil.move(path, os.getcwd())
shutil.rmtree(path)
print("Dataset moved to current directory")


import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path
from collections import Counter
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model

print("--- Imports completed ---")
print(f"TensorFlow Version: {tf.__version__}")