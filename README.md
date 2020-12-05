# Speech-Recognition_mod
My mod of Speech-Recognition  

## Original sources  
* https://github.com/iamlekh/Speech-Recognition  

## Dependencies  
* librosa  
calculate mfcc    
* tensorflow-cpu  
deep learning model traning and recognition, using tf.keras    

## How to run for Baidu AIStudio   
```
$ pip install tensorflow-cpu  
$ python  
import tensorflow as tf  
tf.__version__  
exit()  
$ cd  
$ wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz  
$ mkdir ./speech_commands  
$ tar xzf speech_commands_v0.01.tar.gz -C ./speech_commands   
$ mv Speech-Recognition_mod Speech-Recognition  
$ cd Speech-Recognition  
$ cd local  
$ python data_preparation.py  
(run about 13 minutes, generate data.json and label_data.json)  
$ cat label_data.json  
$ pip install tensorflow-cpu  
$ pip list  
tensorflow-cpu         2.3.1  
$ python model_training.py    
Total params: 36,063  
Trainable params: 35,807  
Non-trainable params: 256  
Epoch 1/50  
853/853 - 46s 54ms/step - loss: 2.4780 - accuracy: 0.3033 - val_loss: 1.5747 - val_accuracy: 0.5276  
134/134 - 3s 22ms/step - loss: 0.4158 - accuracy: 0.9055  
Test loss: 0.4157963693141937, test accuracy: 90.54656624794006  
(run about 35 minutes, generate modelf.h5)  
$ cd ../server  
$ cp ../local/model.h5 .  
$ cp ../local/label_data.json .  
$ python keyword_spotting_service.py tests/down.wav   
```

## Original README  
1) local/classifier/data_preparation.py -> to prepare the data 

Data Source - https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html

2) local/classifier/model_training.py -> to build the CNN model and train

3) server/flask/model.h5 -> model

4) server/flask/keyword_spotting_service.py -> to make predictions

5) server/flask/server.py -> Flask app

* to run git clone
* server/init.sh

Ref of the proj-> https://www.youtube.com/playlist?list=PL-wATfeyAMNpCRQkKgtOZU_ykXc63oyzp 
