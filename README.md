# Disaster Response Pipeline Project

## Description

This project was created as part of Udacity's [Data Scientist](https://www.udacity.com/school/data-science) nanodegree program. The goal is to analyze disaster data from [Appen](https://www.figure-eight.com) (formerly Figure 8) and use natural language processing and machine learning techniques to build a model that categorizes messages sent during disaster events so that they can be sent to the appropriate disaster relief agency.

The project contains three components:
- An ETL pipeline that extracts disaster message and category data from `.csv` files, merges and cleans the data, and stores it into an SQLite database
- An ML pipeline that loads the cleaned disaster message data from an SQLite database, generates train/test data sets, builds an NLP machine learning model, trains and fine-tunes the model using GridSearchCV, and exports the final model as a pickle file
- A Flask web app that allows you to enter messages and view its categories according to the model

## Files
The project contains three folders:
- `data` contains an ETL script named `process_data.py`; data files `disaster_categories.csv` and `disaster_messages.csv`; the output SQLite database `DisasterResponse.db`; and `ETL Pipeline Preparation.ipynb`
- `models` contains an NLP/ML pipeline script named `train_classifier.py`, the output model `classifier.pkl`, and `ML Pipeline Preparation.ipynb`
- `app` contains the Flask web app defined in `run.py` along with its html templates

## Instructions:
**Note:** Project was built with *Python 3.6.10*. A `requirements.txt` file containing dependencies is included.

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to [127.0.0.1:3000](http://127.0.0.1:3000/) open the homepage

## License
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Acknowledgements
Thanks to [Udacity](https://www.udacity.com/) for bringing this nanodegree program to the workforce. This course has provided me an outline on executing ETL and ML pipelines that has led to the completion of this project. 

Thanks to [Appen](https://www.figure-eight.com) (formerly Figure 8) for providing this dataset as an opportunity to practice data science and machine learning on real-world data.

