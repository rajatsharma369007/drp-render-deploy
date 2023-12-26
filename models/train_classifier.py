import sys
from sqlalchemy import create_engine
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from typing import Tuple, List
from tokenize_kuma import tokenize_kuma

def load_data(database_filepath: str) -> Tuple[pd.Series, pd.DataFrame, List[str]]:
    """
    Loads disaster data from sqlite database

    Parameters
    ----------
    database_filepath : str
        Path to the sqlite database file

    Returns
    -------
    X : pd.Series
        Message data
    y : pd.DataFrame
        Corresponding category classification data
    category_names : list[str]
        List of categories
    """
    #nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql('messages', engine)

    #get model input and output data
    X = df['message']
    y = df.drop(columns=['id','message','original','genre', 'child_alone']) #drop child_alone category because it has all 0s
    
    return X, y, list(y.columns)


def build_model() -> GridSearchCV:
    """
    Builds machine learning Pipeline and GridSearchCV

    Returns
    -------
    GridSearchCV for the classifier model
    """
    #Build pipeline
    pipeline = Pipeline([
        ('text_pipeline', TfidfVectorizer(tokenizer=tokenize_kuma)), #TfidfVectorizer = CountVectorizer->TFidfTransformer
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=5))) #MultiOutputClassifier using RandomForestClassifier estimator
    ])

    #Build GridSearchCV using parameters below
    parameters = {
        #'text_pipeline__max_df' : [0.5, 0.75, 1.0],
        'text_pipeline__min_df' : [5]
    }

    #return GridSearchCV(pipeline, {'text_pipeline__min_df' : [50, 100]}, scoring='f1_micro', cv=2) #Training may take long using current parameters, use this for shorter processing
    return GridSearchCV(pipeline, parameters, scoring='f1_micro', cv=2)


def evaluate_model(model: GridSearchCV, X_test: pd.Series, Y_test: pd.DataFrame, category_names: List[str]):
    """
    Evaluates the best model of GridSearchCV after fitting

    Parameters
    ----------
    model : GridSearchCV
        The GridSearchCV containing the model
    X_test : pd.Series
        The X values of the test set
    Y_test : pd.DataFrame
        The true Y values of the test set
    category_names : List[str]
        The category names in Y
    """
    y_pred = model.best_estimator_.predict(X_test)
    for i, category in enumerate(category_names):
        print(category_names[i] + '\n' + classification_report(Y_test.iloc[:,i], y_pred[:,i]))
        print(confusion_matrix(Y_test.iloc[:,i], y_pred[:,i]))


def save_model(model: GridSearchCV, model_filepath: str):
    """
    Saves the best model to a pickle file

    Parameters
    ----------
    model : GridSearchCV
        The GridSearchCV containing the model
    model_filepath : str
        File path to save to
    """
    import pickle
    with open(model_filepath, mode='wb') as file:
        pickle.dump(model.best_estimator_, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()