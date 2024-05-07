import argparse
import os
import shutil
import tempfile
import json
import time



import mlflow
import mlflow.sklearn
from azureml.core import Run

import mltable

import pandas as pd
#from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.compose import ColumnTransformer
#from sklearn.pipeline import Pipeline


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", type=str, help="Path to training data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--target_column_name", type=str, help="Name of target column")
    #parser.add_argument("--exclude_features", type=str, help="Name of columns to exlude from the training set")
    parser.add_argument("--model_base_name", type=str, help="Name of the registered model - list")
    
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")
        

    # parse args
    args = parser.parse_args()

    # return args
    return args


def main(args):
    mlflow.sklearn.autolog()

    current_experiment = Run.get_context().experiment
    tracking_uri = current_experiment.workspace.get_mlflow_tracking_uri()
    print("tracking_uri: {0}".format(tracking_uri))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(current_experiment.name)

    # Read in training data
    print("<<< Reading traning data >>>")
    train_tbl = mltable.load(args.training_data)
    train_df = train_tbl.to_pandas_dataframe()

    print("Extracting X_train, y_train")
    print("all_data cols: {0}".format(train_df.columns))
    y_train = train_df[args.target_column_name]
    X_train = train_df.drop(labels=args.target_column_name, axis="columns")
    """
    print("args.exclude_features:",args.exclude_features)
    if not (args.exclude_features == "NA"):
        exclude_features_list = args.exclude_features.strip('"').split(',')
        X_train = X_train.drop(labels=exclude_features_list, axis="columns")
    """
    print("X_train cols: {0}".format(X_train.columns))

    # Read in test data
    print("<<< Reading test data >>>")
    test_tbl = mltable.load(args.test_data)
    test_df = test_tbl.to_pandas_dataframe()

    print("Extracting X_test, y_test")
    y_test = test_df[args.target_column_name]
    X_test = test_df.drop(labels=args.target_column_name, axis="columns")
    """
    #exclude_features
    if not (args.exclude_features == "NA"):
        X_test = X_test.drop(labels=exclude_features_list, axis="columns")
    """
    print("X_test cols: {0}".format(X_test.columns))

    print("Training model")
    # The estimator can be changed to suit
    #model = LGBMClassifier(n_estimators=5)
    #model = GradientBoostingClassifier(n_estimators=args.n_estimators, learning_rate=args.learning_rate)

    #Assume all string variables are categorical
    #categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    #print("<<< categorical_features >>>")
    #print(categorical_features)

    #categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    #preprocessor = ColumnTransformer(transformers=[('cat', categorical_transformer, categorical_features)])
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05)
    #model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', GradientBoostingClassifier())])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("<<< classification_report >>>")
    print(classification_report(y_test, y_pred))

    # Eli - I think just saving the model with out going through temp shoudl work:
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)

    #suffix = int(time.time())
    #registered_name = "{0}_{1}".format(args.model_base_name, suffix) # this way the model version is always 1
    registered_name = args.model_base_name
    print(f"<<< Registering model as {registered_name} >>>")

    print("<<< Registering via MLFlow >>>")
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name=registered_name,
        artifact_path=registered_name,
    )

    print("<<< Writing JSON >>")
    dict = {"id": "{0}:1".format(registered_name)}
    
    output_path = os.path.join(args.model_info_output_path, "model_info.json")
    with open(output_path, "w") as of:
        json.dump(dict, fp=of)

    # Stop Logging
    mlflow.end_run()

# run script
if __name__ == "__main__":
    # add space in logs
    print("*" * 60)
    print("\n\n")

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
