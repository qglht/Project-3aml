import pickle
import gzip
import numpy as np
import os
import cv2
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
    return create_X_y(train_data_chosen_label)


