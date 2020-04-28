# JiraPredictor
Makes predictions and explains them of when tickets will be finished.

Implementation for the Jira Prediction Assignment of Xccelerated.

## Get Started:

First, clone this repository:

```git clone https://github.com/VictorZuanazzi/JiraPredictor.git ``` 

You will need the following python packages:
* flask
* flask_jsonpify
* numpy
* pandas
* sklearn

If you are using Anaconda, download and isntall the virtual environemnt


```conda env create -f xcc.yml```

Otherwise use pip to install the requirements


```pip install requirements.txt ```

## Run predictions:

There is a trained model available for plug and play use. Just do


```cd  path/to/JiraPredictor```
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

Go , use ISO-8601 or YYYY-MM-DD


```http://127.0.0.1:5000/api/issue/<date>/resolved-since-now```
  
For example:


```http://127.0.0.1:5000/api/issue/2020-07-15T01:31:58.182898/resolved-since-now```
```http://127.0.0.1:5000/api/issue/2020-07-15/resolved-since-now```
 

### Explain predictions:

There is an explanation engine added to the API to help the user with insights, go to:


```http://127.0.0.1:5000/api/issue/<issue_key>/resolve-prediction```


Explain a closed issue:


```http://127.0.0.1:5000/api/issue/AVRO-1/explain-prediction```

Explain an open issue:


```http://127.0.0.1:5000/api/issue/AVRO-2167/explain-prediction```

Have fun!

