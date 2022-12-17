import pickle
import gzip
import numpy as np
import os
import cv2
import ipdb
from typing import List, Tuple

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

def save_zipped_pickle(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)

def select_dataset(data:List[dict], quality:str) -> List[dict]:
    """Selects a dataset from the initial data

    Args:
        data (List[dict]): initial data (containing amateur and expert)
        quality (str): either 'amateur' or 'expert'

    Returns:
        List[dict]: dataset with only the quality chosen
    """
    if "all":
        return data
    else : 
        return [dict_frames for dict_frames in data if dict_frames["dataset"]==quality]

def select_only_label_images(data:List[dict])->dict:
    """Retrieves only the images with labels from the dataset

    Args:
        data (List[dict]): output from select_dataset

    Returns:
        dict: key: name of the video | value: {"image":image, "label":label}
    """
    images_labelized = {}
    for video in data:
        indices_labelized_frames = video["frames"]
        for index in indices_labelized_frames:
            images_labelized[video["name"]+"_"+str(index)] = {"image":video["video"][:,:,index], "label":video["label"][:,:,index]}
    return images_labelized

def create_X_y(data_labelized:dict) -> Tuple[np.array, np.array]:
    """Creates two numpy arrays containing the images and labels

    Args:
        data_labelized (dict): output of select_only_label_images

    Returns:
        Tuple[np.array, np.array]: X, y
    """
    X, y = [], []
    for name in data_labelized.keys():
        X.append(data_labelized[name]["image"])
        y.append(data_labelized[name]["label"])
    return np.array(X), np.array(y)

def normalize(X:np.array, y:np.array) -> Tuple[np.array, np.array]:
    """Function to normalize both X and y : resizing and normalizing

    Args:
        X (np.array): images
        y (np.array): labels

    Returns:
        Tuple[np.array, np.array]: X, y normalized and resized
    """
    norm_img = []
    resized_lab = []
    min_shape = 128

    for img in range(X.shape[0]):
        image = X[img]
        label = y[img].astype(float)

        resized_image = cv2.resize(image, (min_shape,min_shape), interpolation=cv2.INTER_LANCZOS4)
        resized_label = cv2.resize(label, (min_shape,min_shape), interpolation=cv2.INTER_LANCZOS4)
        norm_image = cv2.normalize(resized_image,  np.zeros(resized_image.shape), 0, 255, cv2.NORM_MINMAX)

        norm_img.append(norm_image)
        resized_lab.append(resized_label)

    return np.array(norm_img), np.array(resized_lab)
    

def data_treatment(path:str, quality) -> Tuple[np.array, np.array]:
    """Return X and y normalized

    Args:
        path (str)
        quality (str): either "amateur" or "expert"

    Returns:
        Tuple[np.array, np.array]: X,y
    """

    train_data = load_zipped_pickle(path)
    train_data_chosen = select_dataset(train_data, quality)
    train_data_chosen_label = select_only_label_images(train_data_chosen)
    X, y = create_X_y(train_data_chosen_label)
    X_normalized, y_normalized = normalize(X, y)
    return X_normalized, y_normalized


