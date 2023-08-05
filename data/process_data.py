import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads the disaster messages and categories data from two files.

    Args:
        messages_filepath (str): The filepath to the messages CSV file.
        categories_filepath (str): The filepath to the categories CSV file.

    Returns:
        pd.DataFrame: The merged DataFrame of messages and categories.
    """

    message_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    merged_df = pd.merge(message_df, categories_df, how='inner', on='id')
    return merged_df


def clean_data(df):
    """Cleans the disaster messages and categories data.

    Args:
        df (pd.DataFrame): The DataFrame of messages and categories.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """

    df['categories'] = df['categories'].str.split(';')
    df['categories'] = df['categories'].apply(lambda x: dict(s.split('-') for s in x))
    df = pd.concat([df, pd.json_normalize(df['categories'])], axis='columns')
    df = df.drop(labels=['categories'], axis='columns')
    return df


def save_data(df, database_filepath):
    """Saves the cleaned disaster messages and categories data to a SQLite database.

    Args:
        df (pd.DataFrame): The cleaned DataFrame of messages and categories.
        database_filepath (str): The filepath to the SQLite database file.

    Returns:
        None.
    """

    engine = create_engine(f'sqlite:///{database_filepath}', echo=True)
    conn = engine.connect()
    table_name = 'DisasterMessage'
    df.to_sql(table_name, conn, if_exists="replace")
    conn.close()
    return True


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
