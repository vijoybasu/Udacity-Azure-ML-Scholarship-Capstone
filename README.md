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

The results of the AutoML run was the 'Voting Ensemble' model achievening a peak accuracy of `0.87` with below parameters obtained as the best ones.

A voting ensemble (or a “majority voting ensemble“) is an ensemble machine learning model that combines the predictions from multiple other models. It is a technique that may be used to improve model performance, ideally achieving better performance than any single model used in the ensemble.

It is observed that Voting Ensemble uses a logistic regression model with multiple weights and then chooses or "votes" them to select the best one.

```python
min_child_weight=1,
missing=nan,
n_estimators=50,
n_jobs=1,
nthread=None,
objective='reg:logistic',
random_state=0,
reg_alpha=1.875,
reg_lambda=0.20833333333333334,
scale_pos_weight=1,
seed=None,
silent=None,
subsample=0.6,
tree_method='auto',
verbose=-10,
verbosity=0))],
verbose=False))],
flatten_transform=None,
weights=[0.375, 0.125, 0.125,0.125, 0.125,0.125]
```

![image](https://user-images.githubusercontent.com/81923226/115025552-20326000-9edf-11eb-9cf5-4950cc8501bb.png)
![image](https://user-images.githubusercontent.com/81923226/115025758-5bcd2a00-9edf-11eb-975b-7283966f32e8.png)

![image](https://user-images.githubusercontent.com/81923226/115025441-fed17400-9ede-11eb-9de2-55549bcf01d7.png)
#### Confirming the accuracy of the model in the Azure Portal.
![image](https://user-images.githubusercontent.com/81923226/115025599-304a3f80-9edf-11eb-981b-ad696c17eb80.png)

#### RunDetails widget output
![image](https://user-images.githubusercontent.com/81923226/115025876-7dc6ac80-9edf-11eb-9512-8080cddfe8f7.png)


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

The HyperDrive Run with SK Learn's Logistic Regression was able to achieve a better accuracy of `0.92`. 
The model could be further improved by playing around with more hyper-parameters of Logistic Regression such as penalty,class_weight, n_jobs etc.

![image](https://user-images.githubusercontent.com/81923226/115027181-03972780-9ee1-11eb-9e6f-cbba1015c882.png)
![image](https://user-images.githubusercontent.com/81923226/115029797-f9c2f380-9ee3-11eb-91c7-b4f41e96829a.png)

#### Accuracy of 0.92 achieved.
![image](https://user-images.githubusercontent.com/81923226/115027205-0a259f00-9ee1-11eb-8476-f273812c09af.png)
![image](https://user-images.githubusercontent.com/81923226/115027229-13167080-9ee1-11eb-8406-7d8934e62012.png)
![image](https://user-images.githubusercontent.com/81923226/115027278-2295b980-9ee1-11eb-81f7-c3010b32601a.png)
![image](https://user-images.githubusercontent.com/81923226/115027290-275a6d80-9ee1-11eb-900a-afc5a790dc42.png)

#### RunDetails widget output
![image](https://user-images.githubusercontent.com/81923226/115027306-2c1f2180-9ee1-11eb-831c-688da188f4d1.png)
![image](https://user-images.githubusercontent.com/81923226/115027356-3d682e00-9ee1-11eb-90c6-ef99e2127907.png)
![image](https://user-images.githubusercontent.com/81923226/115027370-435e0f00-9ee1-11eb-9845-3b65a12d1fc3.png)


## Model Deployment
We compare and it is clearly seen that the model obtained by the HyperDrive config run obtains an accuracy of `0.92` using the SKLearn's Logistic Regression.

We shall go ahead and deploy this particular model and test the endpoint.

#### Deployment Screenshots as below:

![image](https://user-images.githubusercontent.com/81923226/115027501-6c7e9f80-9ee1-11eb-8e0c-d0468e8158b0.png)
![image](https://user-images.githubusercontent.com/81923226/115027514-70aabd00-9ee1-11eb-8e4a-e990233321e8.png)
![image](https://user-images.githubusercontent.com/81923226/115027520-73a5ad80-9ee1-11eb-81aa-e9e720f066c2.png)
![image](https://user-images.githubusercontent.com/81923226/115027531-77393480-9ee1-11eb-9e05-35379434c542.png)
![image](https://user-images.githubusercontent.com/81923226/115027543-7accbb80-9ee1-11eb-8d01-75f8f11eac66.png)
![image](https://user-images.githubusercontent.com/81923226/115027557-7ef8d900-9ee1-11eb-94ee-25adb7617881.png)
![image](https://user-images.githubusercontent.com/81923226/115027703-ab145a00-9ee1-11eb-8445-26eb13d78e68.png)


#### Testing Model Endpoint

The model endpoint is tested using a sample of row values in the test dataset. The output is obtained as `[1,0]` indicating that the said person will indeed have a death event owing to his physical factors.
![image](https://user-images.githubusercontent.com/81923226/115027967-f9295d80-9ee1-11eb-8972-e9aabce8fd43.png)


## Screen Recording
https://www.youtube.com/watch?v=lyTwnFrMKms

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/lyTwnFrMKms/0.jpg)](https://www.youtube.com/watch?v=lyTwnFrMKms)

## Future Improvements
Benchmarking can be done using Apache Benchmark command-line tool to keep the performance of the model in check and above a standard level. It is used to determine the response time in seconds for the model that is deployed. Additionally, you could try different models to get the best possible one. If we reduced the duration of the experiment or increased the number of processes running in parallel then the experiment will be fast and time can be save - however resource costs may increase. The use of the Kubernetes service can be helpful in case we add more data to the existing dataset. Lastly, the exploration of possibly using Deep Learning to get to a more accurate model. Lastly, it would be even better if we could get more data records from electronic health records. This project took approximately 5 hours to complete.


## ALL SCREENSHOTS
![image](https://user-images.githubusercontent.com/81923226/115028215-3beb3580-9ee2-11eb-8ad0-7b2f103cf35c.png)
![image](https://user-images.githubusercontent.com/81923226/115028228-40175300-9ee2-11eb-92fe-2eb7ae3e86e6.png)
![image](https://user-images.githubusercontent.com/81923226/115028248-44437080-9ee2-11eb-86a4-1cc394f79d30.png)
![image](https://user-images.githubusercontent.com/81923226/115028262-47d6f780-9ee2-11eb-8c8b-0b4457d97358.png)
![image](https://user-images.githubusercontent.com/81923226/115028279-4b6a7e80-9ee2-11eb-89dd-cd748d2cd646.png)
![image](https://user-images.githubusercontent.com/81923226/115028299-502f3280-9ee2-11eb-9e64-54cde62fddbe.png)
![image](https://user-images.githubusercontent.com/81923226/115028313-53c2b980-9ee2-11eb-9b73-ddc4bf2e6acc.png)
![image](https://user-images.githubusercontent.com/81923226/115028331-59200400-9ee2-11eb-802f-90d9066daa95.png)
![image](https://user-images.githubusercontent.com/81923226/115028351-5e7d4e80-9ee2-11eb-8085-e48f1b0a1780.png)
![image](https://user-images.githubusercontent.com/81923226/115028382-65a45c80-9ee2-11eb-84d7-034da46ccd5a.png)
![image](https://user-images.githubusercontent.com/81923226/115028396-6937e380-9ee2-11eb-98bd-4424691d90a8.png)
![image](https://user-images.githubusercontent.com/81923226/115028410-6ccb6a80-9ee2-11eb-898c-a5e59d5c03df.png)
![image](https://user-images.githubusercontent.com/81923226/115028418-7228b500-9ee2-11eb-83e0-03310c3fe993.png)
![image](https://user-images.githubusercontent.com/81923226/115028428-75bc3c00-9ee2-11eb-950b-274e9221ea5c.png)
![image](https://user-images.githubusercontent.com/81923226/115028436-79e85980-9ee2-11eb-861f-badf87b942e5.png)
![image](https://user-images.githubusercontent.com/81923226/115028450-7e147700-9ee2-11eb-9e74-47d75652b743.png)
![image](https://user-images.githubusercontent.com/81923226/115028461-81a7fe00-9ee2-11eb-8a15-7e62de483f73.png)
![image](https://user-images.githubusercontent.com/81923226/115028476-866cb200-9ee2-11eb-9b08-a9bf7c34daa9.png)
![image](https://user-images.githubusercontent.com/81923226/115028488-8a98cf80-9ee2-11eb-8ddf-1ab78b6554f5.png)

### AutoML Run
![image](https://user-images.githubusercontent.com/81923226/115028510-92f10a80-9ee2-11eb-86e7-dc9030e02a76.png)
![image](https://user-images.githubusercontent.com/81923226/115028523-96849180-9ee2-11eb-8bbe-8192b87d02d2.png)
![image](https://user-images.githubusercontent.com/81923226/115028531-9a181880-9ee2-11eb-91c2-54e503426b7b.png)
![image](https://user-images.githubusercontent.com/81923226/115028545-9dab9f80-9ee2-11eb-9679-04eb5bbb566d.png)
![image](https://user-images.githubusercontent.com/81923226/115028558-a13f2680-9ee2-11eb-95bf-c51fc2e98ad6.png)
![image](https://user-images.githubusercontent.com/81923226/115028571-a4d2ad80-9ee2-11eb-81ae-944422fc2085.png)
![image](https://user-images.githubusercontent.com/81923226/115028581-a8663480-9ee2-11eb-8a0b-42fe104da202.png)
![image](https://user-images.githubusercontent.com/81923226/115028606-ac925200-9ee2-11eb-9406-cacf82a1eed4.png)
![image](https://user-images.githubusercontent.com/81923226/115028622-b1570600-9ee2-11eb-8691-e42258f902e7.png)
#### VOTING ENSEMBLE with 0.879 accuracy
![image](https://user-images.githubusercontent.com/81923226/115028643-bb790480-9ee2-11eb-9b90-6d1ecd6fa2b6.png)
![image](https://user-images.githubusercontent.com/81923226/115028660-bfa52200-9ee2-11eb-8b9f-e309f7bf003c.png)
![image](https://user-images.githubusercontent.com/81923226/115028666-c338a900-9ee2-11eb-9cc1-204a3a57fc55.png)
![image](https://user-images.githubusercontent.com/81923226/115028678-c6cc3000-9ee2-11eb-986b-cdf46915236d.png)
![image](https://user-images.githubusercontent.com/81923226/115028694-caf84d80-9ee2-11eb-9e67-e09f26e33991.png)
![image](https://user-images.githubusercontent.com/81923226/115028704-ce8bd480-9ee2-11eb-97f0-1e7d6ab25c88.png)
![image](https://user-images.githubusercontent.com/81923226/115028722-d481b580-9ee2-11eb-8a6f-bddbaf3fd1eb.png)


### HyperDrive Run

![image](https://user-images.githubusercontent.com/81923226/115028794-e8c5b280-9ee2-11eb-9fe6-ffd8dd40cb2a.png)
![image](https://user-images.githubusercontent.com/81923226/115028814-ecf1d000-9ee2-11eb-8867-df77d7a3331e.png)
![image](https://user-images.githubusercontent.com/81923226/115028823-f0855700-9ee2-11eb-890e-7f6873a46d18.png)
![image](https://user-images.githubusercontent.com/81923226/115028842-f4b17480-9ee2-11eb-8a0e-241f0bd43f83.png)
![image](https://user-images.githubusercontent.com/81923226/115028857-f844fb80-9ee2-11eb-9f55-c26c16f65d63.png)
![image](https://user-images.githubusercontent.com/81923226/115028868-fbd88280-9ee2-11eb-9b8a-93dc6ad3fd73.png)
![image](https://user-images.githubusercontent.com/81923226/115028892-009d3680-9ee3-11eb-8f3d-849972e58ac1.png)
#### HyperDrive run achieves 0.92 accuracy
![image](https://user-images.githubusercontent.com/81923226/115028907-05fa8100-9ee3-11eb-8dee-7c5c0e0475b7.png)
![image](https://user-images.githubusercontent.com/81923226/115028918-098e0800-9ee3-11eb-9fd4-e3880443a20b.png)
![image](https://user-images.githubusercontent.com/81923226/115028930-0c88f880-9ee3-11eb-8552-effc6d214c22.png)
![image](https://user-images.githubusercontent.com/81923226/115028942-0f83e900-9ee3-11eb-97ca-35dea58446c4.png)

![image](https://user-images.githubusercontent.com/81923226/115029210-65589100-9ee3-11eb-975f-c2470e03a748.png)
![image](https://user-images.githubusercontent.com/81923226/115029248-6db0cc00-9ee3-11eb-963a-edbc61cba7f0.png)
![image](https://user-images.githubusercontent.com/81923226/115029263-70abbc80-9ee3-11eb-88cd-56c3c255168d.png)
![image](https://user-images.githubusercontent.com/81923226/115029280-74d7da00-9ee3-11eb-8402-cca4bbed9d7a.png)
![image](https://user-images.githubusercontent.com/81923226/115029293-786b6100-9ee3-11eb-8f35-e4c99bb1bf33.png)
![image](https://user-images.githubusercontent.com/81923226/115029304-7b665180-9ee3-11eb-84bc-72bdf9bc4c75.png)
![image](https://user-images.githubusercontent.com/81923226/115029322-7e614200-9ee3-11eb-93a8-ac628eddbcbb.png)
![image](https://user-images.githubusercontent.com/81923226/115029350-84efb980-9ee3-11eb-80ec-3c453673b3df.png)
![image](https://user-images.githubusercontent.com/81923226/115029379-89b46d80-9ee3-11eb-91b9-41e29250afb5.png)

#### Deleting the compute and the model
![image](https://user-images.githubusercontent.com/81923226/115029407-920ca880-9ee3-11eb-89cc-78f2e29aae95.png)

#### Active REST Endpoint
![image](https://user-images.githubusercontent.com/81923226/115029442-9a64e380-9ee3-11eb-9efe-e97bfb2b6467.png)
![image](https://user-images.githubusercontent.com/81923226/115029451-9f299780-9ee3-11eb-9444-63645d53d2cd.png)

