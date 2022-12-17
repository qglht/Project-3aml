import ipdb
from data_treatment import data_treatment, save_zipped_pickle, load_zipped_pickle
from tensorflow import keras
from keras import Model
import numpy as np
from models import jaccard_loss, jaccard_similarity
import cv2
import matplotlib.pyplot as plt

def preprocess_test(data:dict, height_target:int, width_target:int)->list:
    """Transforms video into list of frames

    Args:
        data (dict)
        height_target (int)
        width_target (int)

    Returns:
        list: list of frames
    """
    X_fin = []
    video = data['video']
    for i in range(video.shape[2]):
        frame = cv2.resize(video[:,:,i], (height_target, width_target), interpolation = cv2.INTER_NEAREST)
        X_fin.append(frame)
    return X_fin 

def submit(test_data:list, model:Model, height_unet:int, width_unet:int)->None:
    """Saves the pickle file by making the predictions using the given model

    Args:
        test_data (list)
        model (Model)
        height_unet (int)
        width_unet (int)
    """
    predictions = []
    for data in test_data:
        prediction = np.array(np.zeros_like(data['video']), dtype=np.bool)
        height_target = data["video"].shape[0]
        width_target = data["video"].shape[1]
        video_resized_original = preprocess_test(data, height_target, width_target)
        print(len(video_resized_original))
        for i, frame in enumerate(video_resized_original):
            frame_resized = cv2.resize(frame, (height_unet,width_unet), interpolation=cv2.INTER_NEAREST)
            frame_norm = cv2.normalize(frame_resized,  np.zeros(frame_resized.shape), 0, 255, cv2.NORM_MINMAX)

            pred = model.predict(frame_norm.reshape(1,height_unet, width_unet,1))
            pred = pred > 0.5
            pred = pred * 1
            prediction[:,:,i] = cv2.resize(pred[0,:,:,0], (width_target, height_target ), interpolation = cv2.INTER_NEAREST)
        predictions.append({
            'name': data['name'],
            'prediction': prediction
            }
        ) 
    save_zipped_pickle(predictions, 'my_predictions.pkl')

def submission():
    # be careful, there isn't any label for test data so y_test does not exist
    test_data = load_zipped_pickle("task3/test.pkl")
    model = keras.models.load_model('model_bs_64.h5', custom_objects={"jaccard_loss":jaccard_loss, "jaccard_similarity":jaccard_similarity})
    submit(test_data=test_data, model=model, height_unet=512, width_unet=512)

submission()