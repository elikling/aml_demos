import argparse
import os
import numpy as np
import json

from azureml.core import Run
import mlflow
import mlflow.sklearn
import mltable
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", type=str, help="Path to training data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--target_column_name", type=str, help="Name of target column")
    parser.add_argument("--exclude_features", type=str, help="Name of columns to exlude from the training set")
    
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")
    parser.add_argument("--model_base_name", type=str, help="Name of the registered model")

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
    #X_train = X_train.drop(labels=args.exclude_features, axis="columns")
    X_train[args.exclude_features] = 0 #I cannot drop it so I make it not relevant
    print("X_train cols: {0}".format(X_train.columns))

    # Read in test data
    print("<<< Reading test data >>>")
    test_tbl = mltable.load(args.test_data)
    test_df = test_tbl.to_pandas_dataframe()

    print("Extracting X_test, y_test")
    print("all_data cols: {0}".format(test_df.columns))
    y_test = test_df[args.target_column_name]
    X_test = test_df.drop(labels=args.target_column_name, axis="columns")
    #exclude_features
    #X_test = X_test.drop(labels=args.exclude_features, axis="columns")
    print("X_test cols: {0}".format(X_test.columns))


    print("Training model")
    model = LogisticRegression()
    
    model.fit(X_train, y_train)

    # Print coefficients and intercept
    print(">>>> Coefficients (weights): <<< ")
    print(model.coef_)
    print(">>> Intercept (bias):", model.intercept_)

    y_pred = model.predict(X_test)
    print("<<< classification_report >>>")
    print(classification_report(y_test, y_pred))

    # Compute feature importance (standardized coefficients)
    X_train_reduced = X_train.drop(labels=args.exclude_features, axis="columns")
    print("X_train_reduced cols: {0}".format(X_train_reduced.columns))
    std_X_train = X_train_reduced / np.std(X_train_reduced, axis=0)
    std_model = LogisticRegression()
    std_model.fit(std_X_train, y_train)

    print(">>>> Standardized coefficients: <<<")
    print(std_model.coef_)


    # Eli - I think just saving the model with out going through temp shoudl work:
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)

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
