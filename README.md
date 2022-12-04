# Disaster Response Pipeline Project

This web app is a platform that categorises tweets regarding natural disasters and suggests what type of message the tweet is. This will allow for the correct NGO to act upon the tweet if necessary.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:10049/


(export your PATH as an environment variable)

<<<<<<< HEAD
- app
    - templates
        |- go.html #html for when a request is sent to the app
        |- master.htl #html for rendering the app with graphs on training data
    |- run.py #python file running the app with info on graphs
- data
    |- disaster_categories.csv #data to process
    |- disaster_messages.csv  #data to process
- models
    |- train_classifier.py #sets up pipeline and trains model
    |- classifier.pkl #pickle file containing the trained model
- .gitattributes #allows for large file upload to github
- jv_disast_resp.db #sqlite db file with merged data 
- readme.md

Credit to figure8 for providing the data set.

