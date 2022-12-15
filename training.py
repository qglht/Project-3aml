import ipdb
from data_treatment import data_treatment, augment, visualize
from models import UNET, IoULoss
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras import backend as K
import sklearn
from sklearn.model_selection import train_test_split


def pipeline():
    # data_treatment (choice of the dataset and normalization)
    X_train, y_train = data_treatment("task3/train.pkl", "expert")
    # here do the data augmentation and visualize
    X_train, y_train = augment(X_train, y_train)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
    # data_augmentation
    # model training / tuning
    model = UNET(X_train.shape[1], X_train.shape[2], 1, 64)
    filepath = "model.h5"

    EarlyStop=EarlyStopping(patience=10,restore_best_weights=True)
    Reduce_LR=ReduceLROnPlateau(monitor='val_accuracy',verbose=2,factor=0.5,min_lr=0.00001)
    model_check=ModelCheckpoint('model.h5',monitor='val_loss',verbose=1,save_best_only=True)
    tensorbord=TensorBoard(log_dir='logs')
    callback=[EarlyStop , Reduce_LR,model_check,tensorbord]

    history = model.fit(X_train, y_train, validation_split=0.1, batch_size=32, epochs=100, 
                        callbacks=callback)
    model.save(filepath)
    # evaluation

pipeline()