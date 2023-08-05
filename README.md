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
   `python run.py`

3. Go to http://0.0.0.0:3001/


This will run the project and save the model to a file.
