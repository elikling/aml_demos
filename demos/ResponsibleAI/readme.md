/aml_admin/clear_models.ipynb - clean up the aml-model-regisrty
/demo_data/academic_sucess.csv

Exploration/ResponsibleAI
 - prepare the compute and the data-assets
    - /setup/housing-price-setup.ipynb
    - /setup/academic_sucess_setup.ipynb

 - see piplines folder 


```bash
az login --identity
az account set --subscription "<>"
export AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=true

cd Exploration/ResponsibleAI/pipelines
az ml job create --file housing_cost_pipeline.yaml

az ml job create --file academic_sucess_pipeline.yaml
```