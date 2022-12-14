from keras.metrics import MeanIoU

def evaluation(y_test, y_pred):
    m = MeanIoU(num_classes=2)
    m.update_state(y_test, y_pred)
    return m.result().numpy()