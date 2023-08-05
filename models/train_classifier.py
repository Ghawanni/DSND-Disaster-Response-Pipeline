import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    """Loads the data from the specified database filepath.

    Args:
        database_filepath (str): The filepath of the database file.

    Returns:
        (list, list, list): The messages, categories, and category names.
    """

    conn = create_engine(f'sqlite:///{database_filepath}').connect()
    df = pd.read_sql_table('DisasterMessage', conn)
    df = df.drop(labels=['index', 'id'], axis='columns')
    messages = df.message.values
    categories = df.iloc[:, 3:].values
    category_names = df.iloc[:, 3:].columns
    conn.close()
    return messages, categories, category_names


def tokenize(text):
    """Tokenizes a text string and returns a list of clean tokens.

    Args:
        text (str): The text string to tokenize.

    Returns:
        list: A list of clean tokens.
    """

    # get word tokens
    tokens = word_tokenize(text)

    # Lemmatize every word (token) and remove whitespace and convert to lowercase
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """Builds a machine learning model for multi-label classification.

    Returns:
        GridSearchCV: A grid search object that can be used to fit the model.
    """

    pipeline = Pipeline(
        [('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ("vect", CountVectorizer(tokenizer=tokenize)),
                ("tfidf", TfidfTransformer())
            ]))
        ])),
         ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=2)))
         ]
    )

    # Commented out some params to allow the code to run faster
    parameters = {
        # 'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        # 'clf__n_estimators': [50, 100, 200],
        'clf__estimator__n_estimators': [50],
        # 'clf__min_samples_split': [2, 3, 4],
        # 'clf__max_depth': [5, 10, 20]
        # 'clf__max_depth': [5, 10]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates a machine learning model on a test set.

    Args:
        model (sklearn.model): The machine learning model to evaluate.
        X_test (numpy.ndarray): The test data.
        Y_test (numpy.ndarray): The ground truth labels for the test data.
        category_names (list): The names of the 36 categories.

    Returns:
        None.
    """

    # predict messages category
    y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(y_pred, columns=category_names)
    # transform Y_test to df to loop over it
    Y_test = pd.DataFrame().from_records(Y_test)

    # loop over all categories and print classification_report for each category
    for i in range(len(category_names)):
        print('Category: {}'.format(category_names[i].upper()), "\n\n",
              classification_report(Y_test.iloc[:, i], Y_pred_df.iloc[:, i]))

    print("Best parameters: ", model.best_params_)
    return True


def save_model(model, model_filepath):
    """Saves a machine learning model to a file.

    Args:
        model (sklearn.model): The machine learning model to save.
        model_filepath (str): The filepath to the file where the model will be saved.

    Returns:
        None.
    """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath = 'data/DisasterResponse.db'
        model_filepath = 'models/classifier.pkl'
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
