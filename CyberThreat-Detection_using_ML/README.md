# Information Guide

## 1) Train Option
**Function:** To train and save the model on a specified dataset.<br>
**Info:** The program only supports labelled datasets with threats and normal instances to train the machine learning models. The model assumes the training dataset is highly imbalanced where the normal class count >>> anomaly/ threat class count. The dataset must be in csv format and could have integrated column headers or separate column headers (text file). Certain additional info regarding the dataset such as the label column and the missing value (na) character should also be specified to correctly load and infer data before training the model.

## 2) Train-Test Performance
**Function:** To view the performance of the model on the Testing split (20%) of the Training set.<br>
**Info:** The model is selected from the list of saved models to view the performance metrics.

## 3) Test Option
**Function:** To test the saved models on a specified dataset and obtain performance metrics.<br>
**Info:** The program requires labelled datasets for testing the models. It must also have the same features as the training dataset. The dataset must be in csv format and could have integrated column headers or separate column headers (text file). Certain additional info regarding the dataset such as the missing value (na) character should also be specified to correctly load and infer data before testing the model.

## 4) Threat/ Anomaly Prediction Option
**Function:** To obtain predictions and its visualization using the saved models on a specified dataset.<br>
**Info:** The dataset provided should have the same features as the training dataset and does not require the label column (to be predicted). The dataset must be in csv format and could have integrated column headers or separate column headers (text file). Certain additional info regarding the dataset such as the missing value (na) character should also be specified to correctly load and infer data before making predictions. Visualization can only be called after configuring the input dataset and obtaining the predictions.
