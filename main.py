from data_treatment import data_treatment

def pipeline():
    # data_treatment (choice of the dataset and normalization)
    X_train, y_train = data_treatment("task3/train.pkl", "expert")
    # data_augmentation
    # model training / tuning
    # evaluation

pipeline()