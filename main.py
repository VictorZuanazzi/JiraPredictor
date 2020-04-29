from datetime import datetime
from flask import Flask
from flask_jsonpify import jsonify
import argparse

# my imports
from data_stuff import GeneralAI

# args
parser = argparse.ArgumentParser(description='Create a Jira resolution forecaster')

parser.add_argument('--verbose', action="store_true", default=False,
                    help='Defines if messages will be printed.')
parser.add_argument('--train_new', action="store_true", default=False,
                    help='Uses the pretrained model.')
parser.add_argument('--data_path', type=str, default='./data/',
                    help='path to folder containing the data files. Helper files will also be saved in this folder.')
parser.add_argument("--train_data_file", type=str, default='avro-transitions.csv',
                    help='name of csv file inside data_path containing the training data with the needed features.')
parser.add_argument("--retrieve_data_file", type=str, default='avro-issues.csv',
                    help='name of csv file containing the tickets and the needed features')
parser.add_argument("--labels", type=str, default='labels.csv',
                    help='name of csv file containing the days to resolution of each data point in train_data_file.')
parser.add_argument("--model_path", type=str, default='./model/',
                    help='path to folder containing the models, and where models will be saved.')
parser.add_argument("--model_name", type=str, default='forest.pickle',
                    help='trained model saved as pickle in model_path.')
parser.add_argument("--debug_mode", action="store_true", default=False)
args = parser.parse_args()


# initialize API
app = Flask(__name__)
app.config["DEBUG"] = args.debug_mode

# initialize the forecaster
# it will load the pretrained model if that is saved, otherwise it will train a new one.
forecaster = GeneralAI(verbose=args.verbose,
                       pretrained=not args.train_new,
                       data_path=args.data_path,
                       issues_csv_name=args.retrieve_data_file,
                       transitions_csv_name=args.train_data_file,
                       labels_csv_name=args.labels,
                       model_path=args.model_path,
                       model_name=args.model_name)

# just some dummy issue for testing the API
answer_dict = {'issue': 'AVRO-9999',
               'predicted_resolution_date': '1970-01-01T00:00:00.000+0000'}

# some useful strings for testing
valid_closed_string = '/api/issue/AVRO-1/resolve-prediction'
valid_open_string = '/api/issue/AVRO-2167/resolve-prediction'
hard_coded_string = '/api/issue/AVRO-9999/resolve-prediction'
invalid_string = '/api/issue/AVRO-01/resolve-prediction'
good_date = "/api/issue/2020-07-15T01:31:58.182898/resolved-since-now"


# ### API methods
@app.route('/', methods=['GET'])
def home():
    return "<h1>Awesome JIRA forecaster</h1><p>Works with AI, so it is like, MAGIC!</p>" \
           "<p>Here you can:</p>" \
           "<p>    - Find the estimated day of completion of a ticket: " \
           "http://127.0.0.1:5000/api/issue/issue_key/resolve-prediction</p>" \
           "<p>    - Retrieve all tickets that are estimated to finish before a certain date: " \
           "http://127.0.0.1:5000/api/issue/date/resolved-since-now</p>" \
           "<p>    - Get insights from our explanation engine, it tells you the contribution of different factors." \
           "http://127.0.0.1:5000/api/issue/issue_key/explain-prediction</p>"

@app.route('/api/issue/<issue_key>/resolve-prediction', methods=['GET'])
def prediction(issue_key):
    """Predicted date for the issue,
    :arg
        issue_key: str, unique identifies of the issue.
    :return
        json containing the answer as in the required protocol or 404 error if the issue does not exist."""
    
    # get prediction
    date = forecaster.predict_completion(issue_key)
    
    if date is None:
        # dummy issue
        if issue_key == 'AVRO-9999':
            return jsonify(answer_dict)
        
        # 404 message
        return f"<h1>404</h1><p> The issue key {issue_key} could not be found.</p>", 404
    
    # predicted date
    return jsonify({"issue": issue_key,
                        "predicted_resolution_date": date})


@app.route('/api/issue/<date>/resolved-since-now', methods=['GET'])
def planning(date):
    """Finds the issues that will be finished before the given date,
    :arg
        date: str, string containing the date using either format ISO-8601 or YYYY-MM-DD.
    :return
        json in the required protocol."""
    now = datetime.now().isoformat()
    issues_list = forecaster.resolved_before(date, as_list=True)
    return jsonify({'now': now,
                    'issues': issues_list})


@app.route('/api/issue/<issue_key>/explain-prediction', methods=['GET'])
def explain(issue_key):
    """Explains the prediction of an issue,
    :arg
        issue_key: str, the unique identifier of the issue of interest.
    :return
        json containing the issue key, the predicted date and the contribution of each feature or 404 message if the 
        issue was not found"""
    date = forecaster.predict_completion(issue_key)
    if date is None:
        return f"<h1>404</h1><p> The issue key {issue_key} could not be found.</p>", 404
    else:
        feature_contribution = forecaster.explain_pred(issue_key)
        return jsonify({"issue": issue_key,
                        "predicted_resolution_date": date,
                        "feature contribution": feature_contribution})


app.run()
