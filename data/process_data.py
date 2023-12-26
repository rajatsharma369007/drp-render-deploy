import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """
    Reads in the disaster messages and categories csv files and returns the data as a single `DataFrame` joined on `id`

    Parameters
    ----------
    messages_filepath : `str`
        Path to the disaster messages csv file
    categories_filepath : `str`
        Path to the disaster categories csv file

    Returns
    -------
    `pandas.DataFrame`
        Message and category csv data joined on `id`
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, on='id', how='inner')


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the merged message/category data and replaces the `categories` column with a column for each category with values `{0,1}`. Also drops duplicate messages.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The disaster data loaded from the `load_data` function
    
    Returns
    -------
    `pandas.DataFrame`
        The transformed/cleaned data
    """
    
    #Split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True) #split values in categories column and expand columns
    category_colnames = [cat[:-2] for cat in categories.iloc[0]] #get category names using the first row values (excluding the last two characters)
    categories.columns = category_colnames #rename columns

    #Convert category values to {0,1}
    for column in categories:
        categories[column] = pd.to_numeric(categories[column].str[-1]) #take last character of string and convert to numeric

    categories['related'] = categories['related'].where(categories['related']==0, 1) #fix instances where related=2 to related=1

    #Replace categories column in df with new category columns
    df = df.drop(columns='categories') #drop original categories column in df
    df = pd.concat([df, categories], axis='columns') #concat original df with new categories dataframe

    df = df[df['message']!='#NAME?'] #remove broken message

    #Remove duplicates (by message)
    df = df.drop_duplicates(['message'])
    assert df.duplicated().sum() == 0
    return df

def save_data(df: pd.DataFrame, database_filename: str):
    """
    Saves cleaned disaster message dataset into an sqlite database

    Parameters
    ----------
    df : `pandas.DataFrame`
        The cleaned disaster message and category dataset
    database_filename : `str`
        Path to the sqlite database file
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messages', engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()