#!/bin/bash
az login --identity
az account set --subscription "<>"
export AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=true

az ml model list --name "houseing_cost_pipeline.yaml"
