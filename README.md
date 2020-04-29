# JiraPredictor
Makes predictions and explains them of when tickets will be finished.

Implementation for the Jira Prediction Assignment of Xccelerated.

## Get Started:

First, clone this repository:

```git clone https://github.com/VictorZuanazzi/JiraPredictor.git ``` 

```cd  path/to/JiraPredictor```

You will need the following python packages:
* python 3.8
* flask
* flask_jsonpify
* numpy
* pandas
* sklearn

If you are using Anaconda, download and isntall the virtual environemnt


```conda env create -f environment.yml```

```conda activate xcc```

Otherwise use pip to install the requirements


```pip install requirements.txt ```

Then unzip the data 


```unzip data.zip```

## Run predictions:

There is a trained model available for plug and play use. Just do


```python main.py```


Open your favorite browser and go to


```http://127.0.0.1:5000/```

### Estimate closing date of a ticket:

Go to:


```http://127.0.0.1:5000/api/issue/<issue_key>/resolve-prediction```

For instance, a closed ticket:


```http://127.0.0.1:5000/api/issue/AVRO-1/resolve-prediction```

The example of an open ticket:


```http://127.0.0.1:5000/api/issue/AVRO-2167/resolve-prediction```

### Retrieve all issues that should be finished before a date

Go to:


```http://127.0.0.1:5000/api/issue/<date>/resolved-since-now```
  
Use dates in the format ISO-8601 or YYYY-MM-DD, for example:


```http://127.0.0.1:5000/api/issue/2020-07-15T01:31:58.182898/resolved-since-now```
```http://127.0.0.1:5000/api/issue/2020-07-15/resolved-since-now```
 

### Explain predictions:

There is an explanation engine added to the API to help the user with insights, go to:


```http://127.0.0.1:5000/api/issue/<issue_key>/resolve-prediction```


Explain a closed issue:


```http://127.0.0.1:5000/api/issue/AVRO-1/explain-prediction```

Explain an open issue:


```http://127.0.0.1:5000/api/issue/AVRO-2167/explain-prediction```


## Train a new model

To train a new model is as simple as

```python main.py --train_new```

All the following arguments are optional, use ```-h``` for details on how to use them.
* ```--verbose``` Defines if messages will be printed;
* ```--train_new``` use it if  the pretrained model;
* ```--data_path``` path to folder containing the data files. Helper files will also be saved in this folder.
* ```--train_data_file``` name of csv file inside data_path containing the training data with the needed features. For the assignment this is the avro-transitions.csv. If you want to use another file, make sure that the following fields are included: "key", "status", "from_status", "to_status", "days_in_from_status", "days_since_open", "days_in_current_status", "vote_count", "comment_count", "watch_count", "description_length", "summary_length";
* ```--retrieve_data_file``` name of csv file containing the tickets and the needed features. For the assigment this is the avro-issues.csv;
* ```--labels``` name of csv file containing the days to resolution of each data point in train_data_file. If not given, the labels will be automatically calculated from --train_data_file and saved in data_path as labels.csv;
* ```--model_path``` path to folder containing the models, and where models will be saved;
* ```--model_name``` name of the model to be trained, it is saved as pickle in model_path;
* ```--debug_mode``` start the API in debug mode.




Have fun!

