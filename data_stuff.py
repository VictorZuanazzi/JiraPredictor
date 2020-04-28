import os
from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# my imports
from xutils import load_pickle, save_pickle


class GeneralAI:
    """Class containing all necessary functions to comunicate with the REST API"""

    def __init__(self, verbose=True, pretrained=True, **kwargs):
        """inputs:
            verbose: bool, if False no prints will be executed.
            pretrained: bool, if True it loads a pretrained model and helper files. If no pretrained model is available,
                it trains a new one from scratch.
        kwargs:
            data_path: str, path to folder containing the necessary data.
            issues_csv_name: str, name csv file containing the information necessary for the predictions.
            transitions_csv_name: str, name of csv file containing the data for training the model.
            labels_csv_name: str, name of csv file containing the computed days to completion of past avro issues.
            model_path: str, path to folder containing the trained model.
            model_name: str, name of the model to be used for the predictions.
            """

        self.verbose = verbose
        self.pretrained = pretrained

        # location variables
        data_path = kwargs.get('data_path', './data/')
        issues_csv_name = kwargs.get('issues_csv_name', 'avro-issues.csv')
        transitions_csv_name = kwargs.get('transitions_csv_name', 'avro-transitions.csv')
        labels_csv_name = kwargs.get('label_csv_name', 'labels.csv')

        # # files that are not used:
        # formated_json_name = kwargs.get('formated_json_name', 'formatted-issue.json')
        # issues_json_name = kwargs.get('issues_json_name', 'avro-issues.json')
        # daycounts_csv_name = kwargs.get('daycounts_csv_name', 'avro-daycounts.csv')

        model_path = kwargs.get('model_path', './model/')
        os.makedirs(model_path, exist_ok=True)
        model_name = kwargs.get('model_name', 'forest.pickle')

        # load files
        self.verbalize("Loading files...")
        # self.formated_json_list = read_json(data_path + formated_json_name)
        # self.issues_json_list = read_json(data_path + issues_json_name)
        # self.daycounts_df = pd.read_csv(data_path + daycounts_csv_name)
        self.issues_df = pd.read_csv(data_path + issues_csv_name)
        self.transitions_df = pd.read_csv(data_path + transitions_csv_name)

        if self.pretrained:
            # try to load necessary files and model, if they dont exist the files will pre re-processed and the model
            # will be re-trained.
            try:
                self.labels_df = pd.read_csv(data_path + labels_csv_name)
            except FileNotFoundError as error:
                self.verbalize(f"Labels file not found: {error}.")
                self.verbalize("Making Labels... Running algorithm to make the labels.")
                self.labels_df = self.make_labels()
                self.labels_df.to_csv(data_path + labels_csv_name)

            try:
                self.predictor = load_pickle(model_path + model_name)

            except FileNotFoundError as error:
                self.verbalize(f"Model not found: {error}.")
                self.verbalize("Training model from scratch...")
                self.predictor = self.train_model()
                save_pickle(self.predictor, model_path + model_name)

            # get mean and std
            _ = self.convert_to_model_input()

        else:
            self.verbalize("Making Labels... Sorting out time to completion in the data.")
            self.labels_df = self.make_labels()
            self.labels_df.to_csv(data_path + labels_csv_name)

            # train a super complex AI model that will take over the world.
            self.verbalize("Start training of the model")
            self.predictor = self.train_model()
            save_pickle(self.predictor, model_path + model_name)

    def verbalize(self, message: str = ''):
        """function that prints messages if verbose is set to true.
        :arg
            message: srt, string containing the message to be printed."""
        if self.verbose:
            print(message)

    def convert_to_model_input(self, df=None, train=True):
        """Converts a dataframe containing the necessary fields into the data that can be fed to the model,
        inputs:
            df: pd.DataFrame or None, dataframe containing the data
            train: bool, whether the data is for training or evaluation. That impacts normalization of the data."""

        # convert transitions_df into issues_df
        if df is None:
            df = self.transitions_df
            transitions_closed_df = df[df["status"] == "Closed"].copy()
        else:
            transitions_closed_df = df.copy()
            transitions_closed_df.rename(columns={"status": "to_status"}, inplace=True)

        data_df = pd.DataFrame()

        # first the simple stuff: the numbers
        data_df["vote_count"] = transitions_closed_df["vote_count"]
        data_df['vote_count'].fillna(value=0, inplace=True)
        data_df["comment_count"] = transitions_closed_df["comment_count"]
        data_df['comment_count'].fillna(value=0, inplace=True)
        data_df["watch_count"] = transitions_closed_df["watch_count"]
        data_df['watch_count'].fillna(value=0, inplace=True)
        data_df['description_length'] = transitions_closed_df['description_length']
        data_df['description_length'].fillna(value=0, inplace=True)
        data_df['summary_length'] = transitions_closed_df['description_length']
        data_df['summary_length'].fillna(value=0, inplace=True)

        # convert categoricals into ordered data
        # (that assumes the categoricals are put in the correct order)
        status_mapping = {'': 0, 'Open': 1, 'In Progress': 2, 'Patch Available': 3,
                          'Resolved': 4, 'Reopened': 5, 'Closed': 6}
        data_df["status"] = transitions_closed_df["to_status"].apply(lambda x: status_mapping[x])
        data_df['status'].fillna(value=0, inplace=True)

        # saves the parameters that normalize the train data
        if train:
            self.means = data_df.mean()
            self.stds = data_df.std()

        data_df = (data_df - self.means) / (self.stds + 0.01)

        return data_df

    def make_labels(self):
        '''Uses transitions_csv_name to create the labels for the estimation of completion.
        :returns
            pandas.DataFrame() with only tickets that were closed and a column named
                days_to_close with the labels.'''

        # data needed to make the labels
        data_df = self.transitions_df[["key",
                                       "status",
                                       "from_status",
                                       "to_status",
                                       "days_in_from_status",
                                       "days_since_open"]].copy()

        # some standard data cleaning
        data_df["days_in_from_status"].fillna(value=0.0, inplace=True, axis=0)
        data_df["days_since_open"].fillna(value=0.0, inplace=True, axis=0)
        data_df["from_status"].fillna(value='', inplace=True)

        # select the data that we have labels for:
        data_closed_df = data_df[data_df["status"] == "Closed"]

        # make new column, THE COLUMN
        data_closed_df.loc[:, "days_to_close"] = 0.0

        # calculate the cumulative time to status Closed
        # I am aware there are better ways of processing the data, the choice for the loops is to make sure that there
        # are no mistakes in the label calculation. Which is quite critical.
        for key in data_closed_df["key"]:
            total_time = data_closed_df[data_closed_df['key'] == key][data_closed_df["to_status"] == "Closed"][
                "days_since_open"].values.item()
            elapsed_time = 0
            for idx in data_closed_df[data_closed_df['key'] == key].index:
                days_to_close = total_time - elapsed_time
                elapsed_time += data_closed_df.loc[
                    idx, "days_in_from_status"]
                data_closed_df.loc[idx, "days_to_close"] = days_to_close

        return data_closed_df

    def train_model(self):
        """train a parametized model using the available data,
        for now this model is a simple average given the status"""

        # get data in the correct format
        data_df = self.convert_to_model_input()

        # get the labels
        label_df = self.labels_df["days_to_close"]

        # initialize the model
        model = RandomForestRegressor(verbose=False)

        # hyper parameters for automatic model search
        distributions = {"n_estimators": [5, 10, 100, 1000],
                         "max_depth": [2, 10, None],
                         "min_samples_split": [2, 10, 100],
                         "min_samples_leaf": [1, 10, 20],
                         "max_features": [None, "log2", "sqrt"]
                         }

        # automatic model search using Random Search with Cross Validation
        model_search = RandomizedSearchCV(estimator=model,
                                          param_distributions=distributions,
                                          n_iter=10,
                                          scoring="r2",
                                          cv=3,
                                          refit=True)

        # fit models to the data and select the best one.
        model_search.fit(data_df, label_df)

        return model_search.best_estimator_

    def predict_completion(self, issue_key: str):
        """Uses data in issues_csv to make a prediction,
        :arg
            issue_key: str, the key of the issue in interest.
        :returns
            a date in format ISO-8601"""

        # check if the issue_key is valid
        if not (self.issues_df['key'] == issue_key).any():
            print(f"Key {issue_key} is not valid. Please double check.")
            return None

        # retrieve issue
        issue_status_pd = self.issues_df.loc[self.issues_df['key'] == issue_key]

        # no need to predict the past! Just returns the known end date
        if (issue_status_pd["status"] == "Closed").item():
            return issue_status_pd["resolutiondate"].item()

        # asks the model when it thinks the ticket will be done
        # The model will consider all the variables, the position of the Moon and the mood of Odin.
        issue_status_converted = self.convert_to_model_input(df=issue_status_pd, train=False)
        pred_days = self.predictor.predict(issue_status_converted)

        # convert the prediction into a date
        end_date = self.predict_day_iso(pred_days.item())

        return end_date

    @staticmethod
    def predict_day_iso(pred_days):
        """helper method for converting days into iso-8601"""
        now = datetime.now()
        end_date = now + timedelta(days=pred_days)

        return end_date.isoformat()

    def resolved_before(self, date_str, as_list=True):
        """Uses data in issues_csv to predict which tickes will be done by the given date.
        :arg
            date_str: str, string containing the date using either format ISO-8601 or YYYY-MM-DD.
            as_list: bool, returns a list of dictionaries if set to True.
        :return
            list of dictionaries or dataframe containing the issues that will be finished before the given date."""

        try:
            if len(date_str) == 10:
                date = datetime.strptime(date_str, '%Y-%m-%d')
            elif len(date_str) > 10:
                date_str = date_str[:26]
                date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%f')
        except ValueError as err:
            print(f"[{err}] Could not recoginize date format {date_str}, please use ISO-8601 or YYYY-MM-DD")
            return None

        issues_open_df = self.issues_df[self.issues_df["status"] != "Closed"].copy()
        issues_open_converted_df = self.convert_to_model_input(issues_open_df, train=False)
        issues_open_df["pred_date"] = self.predictor.predict(issues_open_converted_df)

        # converts the pred_date into the datetime format
        now = datetime.now()
        issues_open_df["pred_date_datetime"] = issues_open_df["pred_date"].apply(lambda x: timedelta(days=x) + now)

        # selects issues that will be ready before the date
        issues_pred_close_df = issues_open_df[issues_open_df["pred_date_datetime"] < date][
            ["key", "pred_date_datetime"]]

        # converts the predicted date to iso-8601
        issues_pred_close_df["pred_date_datetime"] = issues_pred_close_df["pred_date_datetime"].apply(
            lambda x: x.isoformat())

        # change column names so the output is as required
        issues_pred_close_df.rename(columns={"key": "issue",
                                             "pred_date_datetime": "predicted_resolution_date"},
                                    inplace=True)
        if as_list:
            # returns a list of dicts
            return issues_pred_close_df.to_dict('records')

        # returns a dataframe
        return issues_pred_close_df

    def explain_pred(self, issue_key):
        """Feature pertubation to explain the prediction
        :arg
            issue_key: str, unique identifier of the issue in interest
        output:
            dictionary containing the features as keys and their normalized contribution to prediction as values."""

        # retrieve issue
        issue_index = self.issues_df['key'] == issue_key
        issue_status_pd = self.issues_df.loc[issue_index]

        # if the issue is closed, then the output is explained by the status
        if (issue_status_pd["status"] == "Closed").item():
            # if the item is closed, then the status is the only thing that explains the 'prediction'
            exp_df = pd.DataFrame(columns=issue_status_pd.columns)
            exp_df.loc[0] = 0.0
            exp_df.loc[0, "status"] = 1.0
            return exp_df.to_dict('records')

        # retrieve all open issues and convert to input format for the model
        issues_open_df = self.issues_df[self.issues_df["status"] != "Closed"].copy()
        issues_open_converted_df = self.convert_to_model_input(issues_open_df, train=False)
        features = issues_open_converted_df.columns

        # makes a dataframe containig the issue 100 times and convert it to input format for the model
        issue_repeated = issue_status_pd.loc[issue_status_pd.index.repeat(100)].copy()
        issue_repeated_converted_df = self.convert_to_model_input(issue_repeated, train=False)

        # make the predition
        issue_pred = self.predictor.predict(issue_repeated_converted_df)

        # pertube one feature per turn and collect how the predictions deviate
        explain_df = pd.DataFrame(columns=features)
        for feature in features:
            temp_df = issue_repeated_converted_df[features].copy()
            temp_df[feature] = np.random.permutation(issues_open_converted_df[feature])[:100]
            preds_ = self.predictor.predict(temp_df)
            explain_df[feature] = np.abs(issue_pred - preds_)

        # normalize the deviations into contributions
        explain_df = explain_df.sum(axis=0) / 100
        explain_df = explain_df.div(explain_df.sum(), axis=0)

        return explain_df.to_dict()


if __name__ == '__main__':
    forecaster = GeneralAI(pretrained=False)
