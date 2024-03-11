# Parallelisation of Senario Modelling (aml componenets)

Senarios modelling suffers from the curse of dimensionality. Very fast the modeler end up with a simulation or optimisation that is complex and evaluating all the scenarios to compare consuming a significant time. It makes sense to run the scenario evaluation in parallel. I demonstrate how the azure machine learning pipelines can be configured for this purpose:
    - Simply run multiple az ml job create
Use the parallel component facility

## Using the bird model to demonstrate parallelisation

*The following will work from within the ./pipelines folder*

**First login to the az facility**

```bash
az login --identity
az account set --subscription "Avanade UK - Data & AI R&D"
export AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=true
```

**running one job**
```bash
az ml job create --file one_bird_model_pipeline.yaml
```

**run three in a go**
```bash
nohup az ml job create --file one_bird_model_10.yaml &
nohup az ml job create --file one_bird_model_75.yaml &
nohup az ml job create --file one_bird_model_90.yaml &
echo "go go go"
```

**run the parallel pipeline**
```bash
az ml job create --file parallel_bird_models_pipeline.yaml
```

## Interesting URLs
- [demo code github](https://github.com/elikling/aml_demos)
- [Hautphenne, S., and Patch, B., (2021). Birth-and-death Processes in Python: The BirDePy Package. arXiv preprint arXiv:2110.05067](https://arxiv.org/abs/2110.05067)
- [BirDePy github](https://birdepy.github.io/)
- [multiprocessing — Process-based parallelism](https://docs.python.org/3/library/multiprocessing.html)
- [az ml job](https://learn.microsoft.com/en-us/cli/azure/ml/job?view=azure-cli-latest)
- [How to debug pipeline reuse issues in Azure Machine Learning?](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-debug-pipeline-reuse-issues?view=azureml-api-2) - Discusses the *Reuse* behavior: Pipleline *ForeceRerun* & Componenet *is_deterministic*
- [Manage inputs and outputs of component and pipeline](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-inputs-outputs-pipeline?view=azureml-api-2&tabs=cli)
- [Create and manage data assets](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-data-assets?view=azureml-api-2&tabs=cli) - see section *Creating data assets from job outputs*
- [How to use parallel job in pipeline (V2)](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-parallel-job-in-pipeline?view=azureml-api-2&tabs=cliv2)