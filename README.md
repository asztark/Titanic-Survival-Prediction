# 🚢 Titanic Survival Prediction

Welcome to the Titanic Survival Prediction Simulator, which uses a logistic regression model to predict whether a passenger would survive the Titanic disaster.
## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Requirements](#data-requirements)
- [Model Evaluation](#model-evaluation)
- [Used Datasets](#used-datasets)

## Overview

This project applies a logistic regression model to predict the survival of passengers aboard the Titanic based on their demographic and ticket information. The model uses the famous Titanic dataset from Kaggle.

## Features

- Preprocesses Titanic data including handling categorical variables and missing values.
- Trains a logistic regression model to predict passenger survival.
- Provides options to evaluate the model performance and make predictions on user-provided data.
- Allows users to input data manually if a CSV file is not provided.

## Installation

To use this project, ensure you have Python 3 and the necessary libraries installed.
```bash
# Install pandas package
pip install pandas
```
```bash
# Install scikit-learn package
pip install scikit-learn
```
## Cloning the repository
```bash
   git clone https://github.com/asztark/Titanic-Survival-Prediction.git
   cd Titanic-Survival-Prediction
```
## Usage 
Note: Please make sure after cloning the repository that 'titanic.tsv' file is located in the same directory as python script. Otherwise the program won't work. <br>
Also, please note that if you provide incorrect data (either in csv file or when prompted by the script) the end result won't be reliable or the program won't work at all. 
```bash
# Run the script using the command line to predict survival:
python titanic_simulator.py --file [path/to/your/data.csv]
```
```bash
# If you don't have a CSV file, you can input your data when prompted:
python titanic_simulator.py
```
### Options
-h, --help <br>
-f, --file: Path to your CSV file containing the input data. Required columns are listed and explained in the program helper or in the [Data Requirements](#data-requirements) section. (optional) <br>
-s, --stats: If included, the script will display evaluation statistics of the model. (optional) <br>
### Examples
```bash
python .\titanic_simulator.py -f test.csv -s
```
```bash
python .\titanic_simulator.py
```
## Data Requirements
When providing a CSV file or inputting data manually, ensure your dataset contains the following columns:<br>
            - Sex: Gender of the passenger (1 for male or 0 for female).<br>
            - Age: Age of the passenger in years.<br>
            - SibSp: Number of siblings or spouses aboard.<br>
            - Parch: Number of parents or children aboard.<br>
            - Fare: Ticket fare paid by the passenger.<br>
            - Pclass_1: 1st passenger class (1 if True, 0 if False).<br>
            - Pclass_2: 2nd passenger class (1 if True, 0 if False).<br>
            - Pclass_3: 3rd passenger class (1 if True, 0 if False).<br>
            - Embarked_C: You boarded in Cherbourg (1 if True, 0 if False).<br>
            - Embarked_Q: You boarded in Queens (1 if True, 0 if False).<br>
            - Embarked_S: You boarded in Southhampton (1 if True, 0 if False).<br>
            - Title_Millitary: You belong in the millitary social class (1 if True, 0 if False).<br>
            - Title_Higher_cls: You belong in the noble social class (1 if True, 0 if False). <br>
            - Title_Mob: Your belong in the commoner social class (1 if True, 0 if False). <br>
### Example of .csv file:
```
Age,Sex,SibSp,Parch,Fare,Pclass_1,Pclass_2,Pclass_3,Embarked_C,Embarked_Q,Embarked_S,Title_Millitary,Title_Higher_cls,Title_Mob
23,1,1,2,14,0,1,0,1,0,0,0,0,1
```
## Model Evaluation
If the --stats flag is used, the script will output:

Precision: The accuracy of positive predictions. <br>
Recall: The proportion of actual positives correctly identified. <br>
F-score: The harmonic mean of precision and recall. <br>
Model Score: Overall accuracy of the model on test data. <br>

## Used Datasets
Dataset used to train and evaluate created model were obtained from kaggle.com website, specifically from "Titanic - Machine Learning from Disaster" project available at https://www.kaggle.com/competitions/titanic
