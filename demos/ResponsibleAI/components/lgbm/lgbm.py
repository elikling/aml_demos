import argparse
import os
import json
import shutil
import tempfile

import mlflow
import mlflow.sklearn
from azureml.core import Run

import mltable

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", type=str, help="Path to training data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--target_column_name", type=str, help="Name of target column")
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

    # Read in data
    print("Reading data")
    tbl = mltable.load(args.training_data)
    all_data = tbl.to_pandas_dataframe()

    print("Extracting X_train, y_train")
    print("all_data cols: {0}".format(all_data.columns))
    y_train = all_data[args.target_column_name]
    X_train = all_data.drop(labels=args.target_column_name, axis="columns")
    print("X_train cols: {0}".format(X_train.columns))

    # Read in test data
    print("<<< Reading test data >>>")
    test_tbl = mltable.load(args.test_data)
    test_df = test_tbl.to_pandas_dataframe()
    y_test = test_df[args.target_column_name]
    X_test = test_df.drop(labels=args.target_column_name, axis="columns")

    print("Training model")
    # The estimator can be changed to suit
    model = LGBMClassifier(n_estimators=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("<<< classification_report >>>")
    print(classification_report(y_test, y_pred))

    # Eli - I think just savign the model with out going through temp shoudl work:
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)

    registered_name = args.model_base_name

    print("Registering via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name=registered_name,
        artifact_path=registered_name,
    )
    
    print("Writing JSON")
    dict = {"id": "{0}:1".format(registered_name)}
    output_path = os.path.join(args.model_info_output_path, "model_info.json")
    with open(output_path, "w") as of:
        json.dump(dict, fp=of)
    
    """
    # Saving model with mlflow - leave this section unchanged
    with tempfile.TemporaryDirectory() as td:
        print("Saving model with MLFlow to temporary directory")
        tmp_output_dir = os.path.join(td, "my_model_dir")
        mlflow.sklearn.save_model(sk_model=model, path=tmp_output_dir)

        print("Copying MLFlow model to output path")
        for file_name in os.listdir(tmp_output_dir):
            print("  Copying: ", file_name)
            # As of Python 3.8, copytree will acquire dirs_exist_ok as
            # an option, removing the need for listdir
            shutil.copy2(src=os.path.join(tmp_output_dir, file_name), dst=os.path.join(args.model_output, file_name))
        """


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
