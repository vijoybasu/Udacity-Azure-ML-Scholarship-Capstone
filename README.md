# Welcome to Vijoy's Repository!

## I'm Passionate about data (small, big, slow, fast) and the insights from data that drive business outcomes

### Udacity Azure Machine Learning Engineer Scholarship - Capstone Project

In this project we are trying to predict the likelihood of the 'DEATH EVENT' in individuals on the basis of their lifestyle choices and other personal health parameters. 

To achieve this, we are making use of the 'heart-failure-clinical-records' dataset from the [UCI ML Repository (click on this link)](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records).

We use two distinct approaches to initiate training of our model, as mentioned below.
1. Using Azure Machine Learning Studio's AutoML to achieve a model with the highest accuracy.
2. Using Azure's Hyperdrive config service on a SKLearn Model to fine-tune parameters and achieve the best model accuracy.

We then compare the two model accuracy's and deploy the best performing model using Azure Container Instances (ACI).

The deployed model is then consumed using a test endpoint.

## Project Set Up and Installation

Heart Failure Dataset from UCI Machine Learning Repository, https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records

This project requires the creation of a compute instance in order to run a Jupyter Notebook & compute cluster to run the Machine Learning Experiments. The dataset has to be manually selected.Two experiments were run using AutoML & HyperDrive. The Best Model Run, Metrics and a deployed model which consumes a RESTful API Webservice in order to interact with the deployed model.

## Dataset

### Overview
Name: heart_failure_clinical_records_dataset.csv

This dataset contains the following features: age, anaemia, creatinine, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, sex, smoking and time. The target is DEATH_EVENT.

Source : [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records)

### Task
This is a classification problem where we will try to predict if the symptoms will cause death in a patient. The target variable is "DEATH_EVENT".

Thirteen (13) clinical features as listed below:

1. age: age of the patient (years)
2. anaemia: decrease of red blood cells or hemoglobin (boolean)
3. high blood pressure: if the patient has hypertension (boolean)
4. creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)
5. diabetes: if the patient has diabetes (boolean)
6. ejection fraction: percentage of blood leaving the heart at each contraction (percentage)
7. platelets: platelets in the blood (kiloplatelets/mL)
8. sex: woman or man (binary)
9. serum creatinine: level of serum creatinine in the blood (mg/dL)
10. serum sodium: level of serum sodium in the blood (mEq/L)
11. smoking: if the patient smokes or not (boolean)
12. time: follow-up period (days)
13. [target] death event: if the patient deceased during the follow-up period (boolean)

### Access

The dataset is accessed via AzureML SDK and is consequently registered in the workspace as below:

```python
from azureml.core.dataset import Dataset
from sklearn.model_selection import train_test_split

key = 'heart-failure-dataset'
if key in ws.datasets.keys():
    dataset = ws.datasets[key]
    print("dataset found!")

else:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv'
    dataset = Dataset.Tabular.from_delimited_files(url)
    dataset = dataset.register(ws,key)

df = dataset.to_pandas_dataframe()
train, test = train_test_split(df, shuffle=True)
train.to_csv('train.csv',index = False)

datastore = ws.get_default_datastore()
datastore.upload_files(files = ['./train.csv'])

train = Dataset.Tabular.from_delimited_files([(datastore,'train.csv')])
```

## Automated ML

We use the primary metric as accuracy here as that is one of the most important factors in a classification tasks. We also enable a 30 minute experiment_timeout to make sure a lot of the compute is not used in training the model. We make sure to also use n_cross_validations as = 3 to make sure our model doesn't over fit.

```python
from azureml.train.automl import AutoMLConfig

# TODO: Put your automl settings here
automl_settings = {"primary_metric":"accuracy", "experiment_timeout_minutes":30, "enable_early_stopping":True, "n_cross_validations":3,"max_concurrent_iterations": 4}

# TODO: Put your automl config here
automl_config = AutoMLConfig(compute_target = compute_target, task = 'classification', training_data = train, label_column_name = 'DEATH_EVENT',**automl_settings)
```

### Results of AutoML Run

The results of the AutoML run was the 'Voting Ensemble' model achievening a peak accuracy of `0.87`

![image](https://user-images.githubusercontent.com/81923226/115025552-20326000-9edf-11eb-9cf5-4950cc8501bb.png)
![image](https://user-images.githubusercontent.com/81923226/115025758-5bcd2a00-9edf-11eb-975b-7283966f32e8.png)

![image](https://user-images.githubusercontent.com/81923226/115025441-fed17400-9ede-11eb-9de2-55549bcf01d7.png)
#### Confirming the accuracy of the model in the Azure Portal.
![image](https://user-images.githubusercontent.com/81923226/115025599-304a3f80-9edf-11eb-981b-ad696c17eb80.png)

#### RunDetails widget output
![image](https://user-images.githubusercontent.com/81923226/115025876-7dc6ac80-9edf-11eb-9512-8080cddfe8f7.png)
![image](https://user-images.githubusercontent.com/81923226/115025895-828b6080-9edf-11eb-9b45-4328fbcb6a4b.png)


## Hyperparameter Tuning

#### Model Chosen : SKLearn's Logistic Regression.

We chose Logistic Regression since it is a well-known binary classifier, and we are indeed dealing with a binary classification problem. A well tuned Logistic Regression has the capability to outperform even the most advanced approaches.

Parameters with Ranges Used:
1) Inverse Of Regularization Strength (C) - (0.01, 0.1, 1, 10, 100)
2) Maximum Iterations/Runs (max_iters) - (50,75,100,125,150,175,200)
3) Algorithm to use in the optimization problem.(solver) - ('liblinear','sag','lbfgs', 'saga')

```python
import shutil
from azureml.core import Environment, ScriptRunConfig 
# TODO: Create an early termination policy. This is not required if you are using Bayesian sampling.
early_termination_policy = BanditPolicy(evaluation_interval=2,slack_factor=0.2)

#TODO: Create the different params that you will be using during training
param_sampling = RandomParameterSampling({'C': choice(0.01, 0.1, 1, 10, 100),
                                        'max_iter' : choice(50,75,100,125,150,175,200),
                                        'solver' : choice('liblinear','sag','lbfgs', 'saga')})

if "training" not in os.listdir():
    os.mkdir("./training")
shutil.copy('./train.py','./training')

sklearn_env = Environment.from_conda_specification(name="sklearn-env",file_path="./conda_dependencies.yaml")
#TODO: Create your estimator and hyperdrive config
estimator = ScriptRunConfig(source_directory='./training', compute_target = compute_target,script= 'train.py',environment= sklearn_env)

hyperdrive_run_config = HyperDriveConfig(run_config=estimator, hyperparameter_sampling=param_sampling,policy=early_termination_policy,max_total_runs=25,
                                    max_duration_minutes=30,
                                    primary_metric_name='Accuracy',
                                    primary_metric_goal=PrimaryMetricGoal.MAXIMIZE)
```

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

The HyperDrive Run with SK Learn's Logistic Regression was able to achieve a better accuracy of `0.92`

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Future Improvements
Benchmarking can be done using Apache Benchmark command-line tool to keep the performance of the model in check and above a standard level. It is used to determine the response time in seconds for the model that is deployed. Additionally, you could try different models to get the best possible one. If we reduced the duration of the experiment or increased the number of processes running in parallel then the experiment will be fast and time can be save - however resource costs may increase. The use of the Kubernetes service can be helpful in case we add more data to the existing dataset. Lastly, the exploration of possibly using Deep Learning to get to a more accurate model. Lastly, it would be even better if we could get more data records from electronic health records. This project took approximately 5 hours to complete.
