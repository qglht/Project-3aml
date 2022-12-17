import ipdb
from data_treatment import data_treatment, labelize_image, visualize, augment
from models import UNET, IoULoss
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras import backend as K
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
from models import jaccard_loss, jaccard_similarity
from evaluation import evaluation


def pipeline():
    # data_treatment (choice of the dataset and normalization)
    X_test, y_test = data_treatment("task3/cropped_validation.pkl", "expert")
    X_test, y_test = augment(X_test, y_test)
    # data_augmentation
    # model training / tuning
    model = keras.models.load_model('model_bs_64.h5', custom_objects={"jaccard_loss":jaccard_loss, "jaccard_similarity":jaccard_similarity})
    y_pred = model.predict(X_test)
    y_pred = y_pred[:,:,:,0]
    print(f"Score IoU : {evaluation(y_test, y_pred)}")

    # evaluation

pipeline()
