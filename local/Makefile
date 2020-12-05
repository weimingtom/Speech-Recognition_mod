PIP := pip
PYTHON := python
RM := rm -rf

all:
	$(PIP) install tensorflow-cpu  
	$(PIP) install librosa  
	$(PYTHON) data_preparation.py    
	$(PYTHON) model_training.py  

clean:
	$(RM) data.json label_daat.json model.h5
	$(RM) *.json *.h5
