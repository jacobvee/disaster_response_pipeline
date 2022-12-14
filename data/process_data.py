import sys
import os
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine

os.chdir('/Users/Veerapen/disaster_response_pipeline')


def load_data(messages_filepath,categories_filepath):
    '''load in data/merge dataframes'''
    messages = pd.read_csv(messages_filepath)
    print('loaded messages')
    categories = pd.read_csv(categories_filepath)
    print('loaded categories')
    df = messages.merge(categories, on='id', how='left')
    categories = df.categories.str.split(";",expand=True)
    row = categories.iloc[:1].squeeze()
    row.tolist()
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames
    for column in categories: 
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)
    categories = categories.replace(2,1)
    df = df.drop(['categories'],axis=1)
    df = pd.concat([df,categories],axis=1)
    return(df)

def clean_data(df):
    '''Remove any duplicate rows from the dataframe so that the model does not become biased.'''
    df = df.drop_duplicates()
    return(df)


def save_data(df, database_filename):
    '''save data to a sqlite db'''
    engine = create_engine('sqlite:///jv_disast_resp.db')
    df.to_sql('disaster_categories_messages', engine, index=False, if_exists = 'replace')



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