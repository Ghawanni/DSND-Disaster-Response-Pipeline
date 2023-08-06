# Disaster Response Pipeline Project

This repository contains the code for a project that performs ETL operations on disaster recovery messages from appen.com and creates a randomforestclassifier model to be able to predict the category of the input message from 36 different categories.

Ultimately, this machine learning application can help corporations to improve their disaster response capabilities. By identifying and prioritizing urgent messages, categorizing messages for efficient routing, and extracting information from messages, corporations can better understand the situation and respond more effectively.
one of the biggest issues that these coporates face is **noise-to-signal ratio** which can be significantly improved by having as a pipeline to respond quickly to relevant messages.

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


### Project Structure
```bash
├── README.md # This readme file
├── app # Flask Web Application Application directory
│   ├── run.py # Flask application run script, this is where the webapp loads the data & model and serves HTTP requests
│   └── templates # Directory containing HTML (handlebars) templates for rendering
│       ├── go.html # HTML component template that gets rendered when a query is submitted for classification
│       └── master.html # Main HTML template
├── data # Data wrangling module directory 
│   ├── DisasterResponse.db # (after running) SQLite DB containing cleaned data
│   ├── disaster_categories.csv # Raw comma-separated category data from Appen.com
│   ├── disaster_messages.csv # Raw comma-separated message data from Appen.com
│   └── process_data.py # Python script responsible for extracting messages,transforming, cleaning for training, and loading to SQLite DB 
├── models # Directory containing the training and model evaluation scripts
│   ├── classifier.pkl # (after running) Pickle file containing the classification model
│   └── train_classifier.py # Pythong script responsible for loading the data from SQLite DB, building, training, evaluating, adn saving the model
├── notebook.ipynb # Data science notebook used for data exploration and development
└── requirements.txt # Dependencies for the project to run
```

<details>
<summary>Results (for each category)</summary>
<p>

Evaluating model...
Category: RELATED

               precision    recall  f1-score   support

           0       0.00      0.00      0.00      1174
           1       0.77      1.00      0.87      4061
           2       0.00      0.00      0.00        43

    accuracy                           0.77      5278
macro avg       0.26      0.33      0.29      5278
weighted avg       0.59      0.77      0.67      5278




Category: REQUEST

               precision    recall  f1-score   support

           0       0.83      1.00      0.91      4366
           1       0.00      0.00      0.00       912

    accuracy                           0.83      5278
macro avg       0.41      0.50      0.45      5278
weighted avg       0.68      0.83      0.75      5278



Category: OFFER

               precision    recall  f1-score   support

           0       1.00      1.00      1.00      5259
           1       0.00      0.00      0.00        19

    accuracy                           1.00      5278
macro avg       0.50      0.50      0.50      5278
weighted avg       0.99      1.00      0.99      5278


Category: AID_RELATED

               precision    recall  f1-score   support

           0       0.58      1.00      0.73      3047
           1       1.00      0.00      0.01      2231

    accuracy                           0.58      5278
macro avg       0.79      0.50      0.37      5278
weighted avg       0.76      0.58      0.43      5278




Category: MEDICAL_HELP

               precision    recall  f1-score   support

           0       0.92      1.00      0.96      4838
           1       0.00      0.00      0.00       440

    accuracy                           0.92      5278
macro avg       0.46      0.50      0.48      5278
weighted avg       0.84      0.92      0.88      5278




Category: MEDICAL_PRODUCTS

               precision    recall  f1-score   support

           0       0.95      1.00      0.97      4997
           1       0.00      0.00      0.00       281

    accuracy                           0.95      5278
macro avg       0.47      0.50      0.49      5278
weighted avg       0.90      0.95      0.92      5278




Category: SEARCH_AND_RESCUE

               precision    recall  f1-score   support

           0       0.97      1.00      0.99      5127
           1       0.00      0.00      0.00       151

    accuracy                           0.97      5278
macro avg       0.49      0.50      0.49      5278
weighted avg       0.94      0.97      0.96      5278



Category: SECURITY

               precision    recall  f1-score   support

           0       0.98      1.00      0.99      5176
           1       0.00      0.00      0.00       102

    accuracy                           0.98      5278
macro avg       0.49      0.50      0.50      5278
weighted avg       0.96      0.98      0.97      5278




Category: MILITARY

               precision    recall  f1-score   support

           0       0.96      1.00      0.98      5076
           1       0.00      0.00      0.00       202

    accuracy                           0.96      5278
macro avg       0.48      0.50      0.49      5278
weighted avg       0.92      0.96      0.94      5278


Category: CHILD_ALONE

               precision    recall  f1-score   support

           0       1.00      1.00      1.00      5278

    accuracy                           1.00      5278
macro avg       1.00      1.00      1.00      5278
weighted avg       1.00      1.00      1.00      5278



Category: WATER

               precision    recall  f1-score   support

           0       0.94      1.00      0.97      4941
           1       0.00      0.00      0.00       337

    accuracy                           0.94      5278
macro avg       0.47      0.50      0.48      5278
weighted avg       0.88      0.94      0.91      5278




Category: FOOD

               precision    recall  f1-score   support

           0       0.89      1.00      0.94      4672
           1       0.00      0.00      0.00       606

    accuracy                           0.89      5278
macro avg       0.44      0.50      0.47      5278
weighted avg       0.78      0.89      0.83      5278





Category: SHELTER

               precision    recall  f1-score   support

           0       0.91      1.00      0.95      4803
           1       0.00      0.00      0.00       475

    accuracy                           0.91      5278
macro avg       0.46      0.50      0.48      5278
weighted avg       0.83      0.91      0.87      5278



Category: CLOTHING

               precision    recall  f1-score   support

           0       0.98      1.00      0.99      5183
           1       0.00      0.00      0.00        95

    accuracy                           0.98      5278
macro avg       0.49      0.50      0.50      5278
weighted avg       0.96      0.98      0.97      5278





Category: MONEY

               precision    recall  f1-score   support

           0       0.98      1.00      0.99      5157
           1       0.00      0.00      0.00       121

    accuracy                           0.98      5278
macro avg       0.49      0.50      0.49      5278
weighted avg       0.95      0.98      0.97      5278



Category: MISSING_PEOPLE

               precision    recall  f1-score   support

           0       0.99      1.00      0.99      5215
           1       0.00      0.00      0.00        63

    accuracy                           0.99      5278
macro avg       0.49      0.50      0.50      5278
weighted avg       0.98      0.99      0.98      5278





Category: REFUGEES

               precision    recall  f1-score   support

           0       0.96      1.00      0.98      5092
           1       0.00      0.00      0.00       186

    accuracy                           0.96      5278
macro avg       0.48      0.50      0.49      5278
weighted avg       0.93      0.96      0.95      5278



Category: DEATH

               precision    recall  f1-score   support

           0       0.96      1.00      0.98      5041
           1       0.00      0.00      0.00       237

    accuracy                           0.96      5278
macro avg       0.48      0.50      0.49      5278
weighted avg       0.91      0.96      0.93      5278





Category: OTHER_AID

               precision    recall  f1-score   support

           0       0.87      1.00      0.93      4618
           1       0.00      0.00      0.00       660

    accuracy                           0.87      5278
macro avg       0.44      0.50      0.47      5278
weighted avg       0.77      0.87      0.82      5278




Category: INFRASTRUCTURE_RELATED

               precision    recall  f1-score   support

           0       0.94      1.00      0.97      4938
           1       0.00      0.00      0.00       340

    accuracy                           0.94      5278
macro avg       0.47      0.50      0.48      5278
weighted avg       0.88      0.94      0.90      5278



Category: TRANSPORT

               precision    recall  f1-score   support

           0       0.95      1.00      0.98      5037
           1       0.00      0.00      0.00       241

    accuracy                           0.95      5278
macro avg       0.48      0.50      0.49      5278
weighted avg       0.91      0.95      0.93      5278




Category: BUILDINGS

               precision    recall  f1-score   support

           0       0.95      1.00      0.98      5023
           1       0.00      0.00      0.00       255

    accuracy                           0.95      5278
macro avg       0.48      0.50      0.49      5278
weighted avg       0.91      0.95      0.93      5278





Category: ELECTRICITY

               precision    recall  f1-score   support

           0       0.98      1.00      0.99      5184
           1       0.00      0.00      0.00        94

    accuracy                           0.98      5278
macro avg       0.49      0.50      0.50      5278
weighted avg       0.96      0.98      0.97      5278


Category: TOOLS

               precision    recall  f1-score   support

           0       1.00      1.00      1.00      5257
           1       0.00      0.00      0.00        21

    accuracy                           1.00      5278
macro avg       0.50      0.50      0.50      5278
weighted avg       0.99      1.00      0.99      5278



Category: HOSPITALS

               precision    recall  f1-score   support

           0       0.99      1.00      0.99      5221
           1       0.00      0.00      0.00        57

    accuracy                           0.99      5278
macro avg       0.49      0.50      0.50      5278
weighted avg       0.98      0.99      0.98      5278


Category: SHOPS

               precision    recall  f1-score   support

           0       1.00      1.00      1.00      5256
           1       0.00      0.00      0.00        22

    accuracy                           1.00      5278
macro avg       0.50      0.50      0.50      5278
weighted avg       0.99      1.00      0.99      5278


Category: AID_CENTERS

               precision    recall  f1-score   support

           0       0.99      1.00      0.99      5224
           1       0.00      0.00      0.00        54

    accuracy                           0.99      5278
macro avg       0.49      0.50      0.50      5278
weighted avg       0.98      0.99      0.98      5278


Category: OTHER_INFRASTRUCTURE

               precision    recall  f1-score   support

           0       0.96      1.00      0.98      5044
           1       0.00      0.00      0.00       234

    accuracy                           0.96      5278
macro avg       0.48      0.50      0.49      5278
weighted avg       0.91      0.96      0.93      5278


Category: WEATHER_RELATED

               precision    recall  f1-score   support

           0       0.73      1.00      0.84      3829
           1       1.00      0.00      0.01      1449

    accuracy                           0.73      5278
macro avg       0.86      0.50      0.42      5278
weighted avg       0.80      0.73      0.61      5278


Category: FLOODS

               precision    recall  f1-score   support

           0       0.92      1.00      0.96      4837
           1       0.00      0.00      0.00       441

    accuracy                           0.92      5278
macro avg       0.46      0.50      0.48      5278
weighted avg       0.84      0.92      0.88      5278


Category: STORM

               precision    recall  f1-score   support

           0       0.90      1.00      0.95      4776
           1       0.00      0.00      0.00       502

    accuracy                           0.90      5278
macro avg       0.45      0.50      0.48      5278
weighted avg       0.82      0.90      0.86      5278


Category: FIRE

               precision    recall  f1-score   support

           0       0.99      1.00      1.00      5230
           1       0.00      0.00      0.00        48

    accuracy                           0.99      5278
macro avg       0.50      0.50      0.50      5278
weighted avg       0.98      0.99      0.99      5278


Category: EARTHQUAKE

               precision    recall  f1-score   support

           0       0.92      1.00      0.96      4833
           1       0.00      0.00      0.00       445

    accuracy                           0.92      5278
macro avg       0.46      0.50      0.48      5278
weighted avg       0.84      0.92      0.88      5278


Category: COLD

               precision    recall  f1-score   support

           0       0.98      1.00      0.99      5156
           1       0.00      0.00      0.00       122

    accuracy                           0.98      5278
macro avg       0.49      0.50      0.49      5278
weighted avg       0.95      0.98      0.97      5278


Category: OTHER_WEATHER

               precision    recall  f1-score   support

           0       0.95      1.00      0.97      5008
           1       0.00      0.00      0.00       270

    accuracy                           0.95      5278
macro avg       0.47      0.50      0.49      5278
weighted avg       0.90      0.95      0.92      5278


Category: DIRECT_REPORT

               precision    recall  f1-score   support

           0       0.80      1.00      0.89      4225
           1       0.00      0.00      0.00      1053

    accuracy                           0.80      5278
macro avg       0.40      0.50      0.44      5278
weighted avg       0.64      0.80      0.71      5278

Best parameters:  {'clf__estimator__max_depth': 5, 'clf__estimator__min_samples_split': 2, 'clf__estimator__n_estimators': 200, 'features__text_pipeline__vect__ngram_range': (1, 2)}
Saving model...
MODEL: models/classifier.pkl
Trained model saved!
</p>
</details>