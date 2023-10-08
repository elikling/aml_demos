#!/bin/bash

conda_env_name="aml_demos"

# Check if the Conda environment exists
if conda info --envs | grep -q "$conda_env_name"; then
    echo "Conda environment '$conda_env_name' already exists."
else
    # Create the Conda environment
    echo "-----------------------------------------------------"
    echo "-- Creating Conda environment '$conda_env_name'... --"
    echo "-----------------------------------------------------"
    conda env create -f conda_demo_env.yaml --force
    echo "-----------------------------------------------------"
    echo "-- Conda environment '$conda_env_name' created. --"
    echo "-----------------------------------------------------"
fi

echo "Installing kernel"

conda activate aml_demos

python --version > demo_python_version.txt
conda list > demo_conda_list.txt
pip list > demo_pip_list.txt
python -m ipykernel install --user --name aml_demos --display-name "aml_demos"
echo "Conda environment setup successfully."

echo "----------"
echo "-- DONE --"
echo "----------"
