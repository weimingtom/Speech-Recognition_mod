import requests
import sys

# server url
URL = "http://127.0.0.1:5000/predict"

# audio file we'd like to send for predicting keyword
FILE_PATH = "tests/no.wav"

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        FILE_PATH = sys.argv[1]    
    # open files
    file = open(FILE_PATH, "rb")
    # package stuff to send and perform POST request
    values = {"file": (FILE_PATH, file, "audio/wav")}
    response = requests.post(URL, files=values)
    data = response.json()

    print("[client] file: {}".format(FILE_PATH))
    print("[client] Predicted keyword: {}".format(data["keyword"]))

