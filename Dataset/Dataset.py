import tensorflow as tf
import numpy as np
from typing import List

# Basic data structure
class Dataset():
    def __init__(self, name: str, data: tf.Tensor):
        self.data : tf.Tensor = data
        self.name : str = name
        self.size : int = data.shape[0]
        self.samples : int = data.shape[2] # samples

class FBGDataset(Dataset):
    def __init__(self, name: str, data: tf.Tensor,I: tf.Tensor,W: tf.Tensor):
        super(FBGDataset, self).__init__(name, data)
        self.I : tf.Tensor = I
        self.W : tf.Tensor = W
    
class AnsweredDataset(FBGDataset):
    def set_answer(self, answer: tf.Tensor):
        self.answer : tf.Tensor = answer
        return self

class WeightedDataset(AnsweredDataset):
    def set_weight(self, weight: tf.Tensor):
        self.weight : tf.Tensor = weight
        return self



# MearuredDataset

from Dataset.loader import load_folder
from Dataset.AutoFitAnswer import GetFBGAnswer

class MeasurementInfo():
    def __init__(self, folder_path: str,  msg: str = ""):
        self.folder_path : str = folder_path
        self.msg : str = msg

    def get_name(self):
        return "measured {}".format(self.msg)

    def get_dataset(self) -> Dataset:
        return MeasuredDataset.from_measurement(self)

class FBGMeasurementInfo(MeasurementInfo):
    def set_fbg_count(self, fbg_count: int):
        self.fbg_count : int = fbg_count
        self.has_config = False
        self.has_threshold = False
        return self
    
    def set_fbg_config(self, I: List[float], W: List[float]):
        self.I : List[float] = I
        self.W : List[float] = W
        self.has_config = True
        return self

    def set_threshold(self, threshold: float):
        self.threshold : float = threshold 
        self.has_threshold = True
        return self

    def get_name(self):
        return "measured {} fbg {}".format(self.fbg_count, self.msg)

    def get_dataset(self) -> FBGDataset:
        return MeasuredDataset.from_measurement(self)
    

class MeasuredDataset():
    @classmethod
    def from_measurement(self, info : MeasurementInfo) -> Dataset:
        raw_dataset = load_folder(info.folder_path)

        if raw_dataset is tuple:

            raw_data, answer_table = raw_dataset
            data = tf.constant(raw_data, tf.dtypes.float32)
            answer = answer_table[:, :info.fbg_count]

            peaks = tf.constant(np.concatenate([
                answer_table[0, info.fbg_count:info.fbg_count*2, np.newaxis], # I
                answer_table[0, :info.fbg_count, np.newaxis],  # C
                answer_table[0, info.fbg_count*2:, np.newaxis],  # W
            ], axis=1), dtype=tf.dtypes.float32)

            I = tf.constant(peaks[:,0], tf.dtypes.float32)
            W = tf.constant(peaks[:,2], tf.dtypes.float32)

            dataset = AnsweredDataset(info.get_name(), data, I, W)
            dataset.set_answer(answer)

            return dataset

        else:

            data = tf.constant(raw_dataset, tf.dtypes.float32)

            if info is FBGMeasurementInfo:
                print('got fbg measurement')
                fbg_info : FBGMeasurementInfo = info

                if fbg_info.has_config:
                    I = tf.constant(info.I, dtype=tf.dtypes.float32)
                    W = tf.constant(info.W, dtype=tf.dtypes.float32)
                    dataset = FBGDataset(info.get_name(), data, I, W)
                
                elif fbg_info.has_threshold:
                    C, I, W = GetFBGAnswer(data, fbg_info.fbg_count, fbg_info.threshold)
                    dataset = AnsweredDataset(info.get_name(), data, I, W)
                    dataset.set_answer(C)

                    return dataset

                else:
                    return Dataset(info.get_name(), data)

            else:
                return Dataset(info.get_name(), data)

        return dataset
        

