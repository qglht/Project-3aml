import pickle
import gzip
import ipdb
from typing import List

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


list_videos = select_dataset(load_zipped_pickle("task3/train.pkl"), "expert")
list_train = list_videos[:15]
list_validation = list_videos[15:17]
list_test = list_videos[17:]
save_zipped_pickle(list_train, "task3/cropped_train.pkl")
save_zipped_pickle(list_train, "task3/cropped_test.pkl")
save_zipped_pickle(list_validation, "task3/cropped_validation.pkl")