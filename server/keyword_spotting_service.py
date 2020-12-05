import librosa
import tensorflow as tf
import numpy as np
import json
import sys

SAVED_MODEL_PATH = "model.h5"
SAMPLES_TO_CONSIDER = 22050
LABEL_DATA_PATH = "label_data.json"
WAV_PATH = "tests/left.wav"

class _Keyword_Spotting_Service:
    model = None
    _mapping = {}
    _instance = None

    def __init__(self):
        self.load_label_data()

    def load_label_data(self):
        with open(LABEL_DATA_PATH, "r") as fp:
            data = json.load(fp)
        labels = data["labels"]
        words = data["words"]
        for index in range(len(labels)):
            # print("labels[index] == ", labels[index])
            self._mapping[labels[index]] = words[index]
            for i in range(labels[index]):
                if i not in self._mapping.keys():
                    self._mapping[i] = "N/A"
        print("[load_label_data] _mapping == ", self._mapping)

    def predict(self, file_path):
        # extract MFCC
        MFCCs = self.preprocess(file_path)
        # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]
        # get the predicted label
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        return predicted_keyword


    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
        # load audio file
        signal, sample_rate = librosa.load(file_path)
        if len(signal) >= SAMPLES_TO_CONSIDER:
            # ensure consistency of the length of the signal
            signal = signal[:SAMPLES_TO_CONSIDER]
            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                         hop_length=hop_length)
        return MFCCs.T


def Keyword_Spotting_Service():
    # ensure an instance is created only the first time the factory function is called
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    return _Keyword_Spotting_Service._instance




if __name__ == "__main__":

    # create 2 instances of the keyword spotting service
    kss = Keyword_Spotting_Service()
    kss1 = Keyword_Spotting_Service()

    # check that different instances of the keyword spotting service point back to the same object (singleton)
    assert kss is kss1
    print("----- keyword_spotting_service ------")
    # make a prediction
    if len(sys.argv) >= 2:
        WAV_PATH = sys.argv[1]
    keyword = kss.predict(WAV_PATH)
    print("[keyword_spotting_service] WAV_PATH: ", WAV_PATH, ", predict result: ", keyword)
