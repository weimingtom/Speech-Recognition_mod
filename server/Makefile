PIP := pip
PYTHON := python
RM := rm -rf

all:
	$(PIP) install tensorflow-cpu  
	$(PIP) install librosa    
	$(PYTHON) keyword_spotting_service.py  

start:
	$(PYTHON) server.py&

stop:
	-killall -q python
	-killall -q python3

test:
	$(PYTHON) keyword_spotting_service.py tests/down.wav 
