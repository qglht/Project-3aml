import ipdb
from data_treatment import data_treatment, labelize_image, visualize
from models import UNET, IoULoss
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras import backend as K
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
from models import jaccard_loss, jaccard_similarity
from evaluation import evaluation

def submission(path_model, path_data):
    # be careful, there isn't any label for test data so y_test does not exist
    X_test, y_test = data_treatment(path_data, "expert")
    model = keras.models.load_model(path_model, custom_objects={"jaccard_loss":jaccard_loss, "jaccard_similarity":jaccard_similarity})
    y_pred = model.predict(X_test)
    y_pred = y_pred[:,:,:,0]