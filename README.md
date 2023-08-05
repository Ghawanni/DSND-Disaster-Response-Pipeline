# Disaster Response Pipeline Project

This repository contains the code for a project that performs ETL operations on disaster recovery messages from appen.com and creates a randomforestclassifier model to be able to predict the category of the input message from 36 different categories.

### Overview

The project is divided into the following steps:

1. Load the data (_process_data.py_)
2. Clean the data (_process_data.py_)
3. Split the data into train and test sets (_train_classifier.py_)
4. Build the model (_train_classifier.py_)
5. Evaluate the model (_train_classifier.py_)
6. Save the model (_train_classifier.py_)


## Usage

To run the project, you will need to have Python 3 and the following packages installed:

* pandas
* numpy
* sklearn
* flask

Once you have installed the necessary packages, you can run the project by following these steps:

1. Clone the repository.
2. Change directory to the repository.
3. Run the following command:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
      `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
      `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
   `python app/run.py`

3. Go to http://0.0.0.0:3001/

<details>
<summary>Results (for each category)</summary>
<p>

Category: SHOPS

               precision    recall  f1-score   support

           0       1.00      1.00      1.00      5260
           1       0.00      0.00      0.00        18

    accuracy                           1.00      5278
macro avg       0.50      0.50      0.50      5278
weighted avg       0.99      1.00      0.99      5278

Category: AID_CENTERS

               precision    recall  f1-score   support

           0       0.99      1.00      0.99      5216
           1       0.00      0.00      0.00        62

    accuracy                           0.99      5278
macro avg       0.49      0.50      0.50      5278
weighted avg       0.98      0.99      0.98      5278

Category: OTHER_INFRASTRUCTURE

               precision    recall  f1-score   support

           0       0.96      1.00      0.98      5059
           1       0.17      0.00      0.01       219

    accuracy                           0.96      5278
macro avg       0.56      0.50      0.49      5278
weighted avg       0.93      0.96      0.94      5278

Category: WEATHER_RELATED

               precision    recall  f1-score   support

           0       0.87      0.96      0.92      3814
           1       0.87      0.63      0.73      1464

    accuracy                           0.87      5278
macro avg       0.87      0.80      0.82      5278
weighted avg       0.87      0.87      0.86      5278

Category: FLOODS

               precision    recall  f1-score   support

           0       0.95      1.00      0.97      4850
           1       0.88      0.38      0.53       428

    accuracy                           0.95      5278
macro avg       0.91      0.69      0.75      5278
weighted avg       0.94      0.95      0.94      5278

Category: STORM

               precision    recall  f1-score   support

           0       0.95      0.98      0.97      4812
           1       0.76      0.50      0.60       466

    accuracy                           0.94      5278
macro avg       0.86      0.74      0.79      5278
weighted avg       0.94      0.94      0.94      5278

Category: FIRE

               precision    recall  f1-score   support

           0       0.99      1.00      0.99      5219
           1       1.00      0.02      0.03        59

    accuracy                           0.99      5278
macro avg       0.99      0.51      0.51      5278
weighted avg       0.99      0.99      0.98      5278

Category: EARTHQUAKE

               precision    recall  f1-score   support

           0       0.97      0.99      0.98      4771
           1       0.91      0.74      0.81       507

    accuracy                           0.97      5278
macro avg       0.94      0.87      0.90      5278
weighted avg       0.97      0.97      0.97      5278

Category: COLD

               precision    recall  f1-score   support

           0       0.98      1.00      0.99      5169
           1       0.80      0.07      0.13       109

    accuracy                           0.98      5278
macro avg       0.89      0.54      0.56      5278
weighted avg       0.98      0.98      0.97      5278

Category: OTHER_WEATHER

               precision    recall  f1-score   support

           0       0.95      1.00      0.97      5005
           1       0.67      0.02      0.04       273

    accuracy                           0.95      5278
macro avg       0.81      0.51      0.51      5278
weighted avg       0.93      0.95      0.93      5278

Category: DIRECT_REPORT

               precision    recall  f1-score   support

           0       0.88      0.98      0.93      4323
           1       0.80      0.38      0.52       955

    accuracy                           0.87      5278
macro avg       0.84      0.68      0.72      5278
weighted avg       0.86      0.87      0.85      5278
</p>
</details>