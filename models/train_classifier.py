import sys
from sqlalchemy import create_engine
import pickle
import re
import sqlalchemy as sa
import pandas as pd
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
import string
from sklearn.model_selection import GridSearchCV


def load_data():
    '''load data from sqlite db'''
    os.chdir(PATH)
    engine = create_engine('sqlite:///jv_disast_resp.db')
    df = pd.read_sql_table('disaster_categories_messages', engine)
    print(df.head())
    X = df.message
    y = df[df.columns[4:]]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    '''Taken from earlier course code (tokenises text/add in a placeholder for urls'''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''builds model using pipeline specified below. Uses gridsearch to tune hyperparameters'''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('rfc', MultiOutputClassifier(RandomForestClassifier()))])
    parameters = {
    'vect__max_df': (0.5,1.0),
    'tfidf__use_idf': (True, False)}
    cv = GridSearchCV(pipeline, parameters, verbose=True, cv=3, n_jobs=-1)
    return(cv)

def evaluate_model(model, X_test, y_test, category_names):
    '''Uses classification report to return a report with accuracy score etc'''
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=y_test.columns)
    for category in y_test.columns:
        category_upper = category.upper()
        print(f"Category: {category_upper}")
        print(classification_report(y_test[category], y_pred[category]))


def save_model(model, model_filepath):
    '''Saves the trained model for use in the web app'''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()