import librosa
import os
import json
import time

DATASET_PATH = '../../speech_commands'
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050
LABEL_DATA_PATH = "label_data.json"

def prepare_data(dataset_path, json_path, label_data_path, samples_to_consider, n_mfcc= 13, hop_length = 512, n_fft = 2048):
  print("[prepare_data] start time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

  data = {
    "mapping":[],
    'labels': [],
    "MFCCs":[],
    'files':[]
  }

  label_data = {
    'labels': [],
    'words': []    
  }
  
  label_set = ["go", "no", "right", "down", "off", "stop", "up", "left", "yes", "on"]
  print("--- prepare_data ---")
  # with open(label_path, mode="w") as file:
  #   file.write("")
  for i,(dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
    # print(dirpath)
    if dirpath not in dataset_path:
      label = dirpath.split("/")[-1]
      if label in label_set:
        data["mapping"].append(label)
        print("Processing: '{}'".format(label))

        # process all audio files in sub-dir and store MFCCs
        for f in filenames:
          file_path = os.path.join(dirpath, f)

          # load audio file and slice it to ensure length consistency among different files
          signal, sample_rate = librosa.load(file_path)

          # drop audio files with less than pre-decided number of samples
          if len(signal) >= samples_to_consider:
                
            # ensure consistency of the length of the signal
            signal = signal[:samples_to_consider]

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=n_mfcc, n_fft=n_fft,
                                          hop_length=hop_length)

            # store data for analysed track
            data["MFCCs"].append(MFCCs.T.tolist())
            data["labels"].append(i)
            data["files"].append(file_path)
            # print("{}: {}".format(file_path, i))
            
        # with open(label_path, mode="a") as file:
        #   file.write("{}: {}\n".format(i, label))
        label_data["labels"].append(i)
        label_data["words"].append(label)

  # print("ready dump...")
  with open(json_path, "w") as fp:
    # print("prepare_data dump")
    json.dump(data, fp, indent=4)

  with open(label_data_path, "w") as fp:
    json.dump(label_data, fp, indent=4)
  
  print("[prepare_data] end time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))




# prepare_data(dataset_path,json_path,13 ,512, 2048)

if __name__ == "__main__":
    prepare_data(DATASET_PATH, JSON_PATH, LABEL_DATA_PATH, SAMPLES_TO_CONSIDER)
