# Parkinson's Freezing of Gait Prediction


The end goal of this project is to accurately predict freezing of gait (FOG) in patients with Parkinson's.


## Setup

This project is done entirely in Python 3, specifically Python 3.9. To install the necessary packages, run:

```
pip install -r requirements.txt
```

Following this, we can fetch our data from the Kaggle competition. The following is the steps to follow to do so:

1. Go to https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction and accept the terms of the competition.
2. Run the following commands:

```
kaggle competitions download -c parkinsons-fog-prediction -p data/
unzip data/parkinsons-fog-prediction.zip -d data/
```
This is a large dataset, so beware that this will take a long time to finish downloading.


## Steps to run

In order to run this project: run the following command:
```
python3 train.py
```