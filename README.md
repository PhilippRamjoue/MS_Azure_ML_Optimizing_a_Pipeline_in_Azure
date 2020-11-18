# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset is of bank marketing campaign and covers information like the job, education, housing situation etc. The task is to create a model that predicts if a marketing action to acquire a new customer was successful. To get a suitable model, two different ways were chosen. On the one hand, a sklearn LogiticRegression model is trained on the dataset. The optimal parameter set is provided by Azure Hyperdrive.
On the other hand, the Azure AutoML tool is used to receive an optimal model from provided Azure algorithms. As a metric, accuracy was chosen because we want a very accurate model to ensure that future marketing efforts are successful. With this metric, it was possible to compare the trained sklearn model and the provided best AutoML model. The best performing solution is a VotingEnsemble model provided by AutoML.

## Scikit-learn Pipeline
The used sklearn pipeline contains different steps:

 1. __Dataset creation__
    
    The marketing dataset is loaded and saved as a TabularDataset.
 2. __Dataset cleaning__

    A lot of columns in the dataset include categorical data like job, education, contact etc. To provide a dataset with numerical data the original one has to be cleaned (dummy values etc.).
 3. __Train Test set generation__

    For the evaluation of the model performance, the dataset is split in a train and test set.

 4. __Model training with different hyperparameters__

    Training of a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) model with different hyperparameters C (Inverse of regularization strength) and max_iter (Maximum number of iterations taken for the solvers to converge).
  

The hyperparameter tuning is executed with Azure [Hyperdrive package](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive?view=azure-ml-py). This tool provides the possibility to define the parameter search space, the metric and many more. The main task is to automate the hyperparameter search process.

To find the best LogisticRegression model the following Hyperdrive config is used:


```
hyperdrive_config = HyperDriveConfig(estimator=est,
                             hyperparameter_sampling=ps,
                             policy=policy,
                             primary_metric_name="accuracy",
                             primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                             max_total_runs=50,
                             max_concurrent_runs=4)
```

 1. __estimator__
    To train the LogisticRegression model is useful to provide a training script (train.py). This script is executed by an [Estimator](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.estimator.estimator?view=azure-ml-py) object.

    ```
    script_parameter = {
    '--C': 1.0,
    '--max_iter': 100
    }

    est = Estimator(source_directory='.',
                script_params=script_parameter,
                compute_target=cpu_cluster,
                entry_script='train.py',
                pip_packages=['sklearn'])
    ```
 2. __hyperparameter_sampling__
   
    To provide the different parameter spaces for the tuning a 
    [RandomParamterSampling](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.randomparametersampling?view=azure-ml-py) object is used. Random sampling is a good start to explore the different parameter combinations and the range in which good values are possible. The next step would be to refine the search space to improve the results. 

    Other provided [algorithms](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperparametersampling?view=azure-ml-py) are Grid and Bayesian sampling. Grip sampling only works with discrete hyperparameters and is not suitable for my purposes because I have no good feeling with discrete C values to use. Additionally, Grid sampling could be very exhaustive. In the Bayesian sampling, the current samples are picked concerning the previous set and the performance. Because of this, the best results are achieved with a small number of concurrent runs. This could lead to an exhaustive and time-consuming tuning task.


    ```
    ps = RandomParameterSampling( {
            '--C': uniform(0.1, 1.0),
            '--max_iter': choice(10, 25, 50, 100, 1000)
        }
    )
    
    ```

    The Object uses uniform distributed values in the range 0.1 to 1.0 for the C parameter and fixed values of choice for the maximum iterations for the solver.



 3. __policy__

    As an early terminating policy, the [BanditPolicy](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py&preserve-view=true#&preserve-view=truedefinition) is chosen. With a small slack_factor, the policy is very aggressive and can save computing time. This is great for the start. 

    Other provided [policies](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#early-termination) are Median stopping policy, Truncation selection policy and No termination policy (default). Median stopping can be used as a conservative policy and Truncation selection as an aggressive policy. No termination policy is no option because we want to save compute time.

    
    ``` 
    policy = BanditPolicy(slack_factor = 0.1, evaluation_interval=1, delay_evaluation=5)
    ```

    The early termination policy is applied at every interval when metrics are reported, starting at evaluation interval 5. Any run whose best metric is less than (1/(1+0.1) or 91% of the best performing run will be terminated.


 4. __primary_metric_name__
   
    The primary_metric_name is provided by the training script:

    
    ```
    train.py

    ...
    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)

    run.log("accuracy", np.float(accuracy))

    ...    
    ```


 5. __primary_metric_goal__
   
    The [primary_metric_goal](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.primarymetricgoal?view=azure-ml-py) is either MAXIMIZE or MINIMIZE. We want a really high accuracy because of this we want to maximize the score.

 6. __max_total_runs__

    The amount of how often the hyperparameter task is executed.

 7. __max_concurrent_runs__
   
    The number of runs that should be executed in parallel. This also depends on the compute cluster and the maximum number of available nodes. 


## AutoML
Before executing the AutoML run some pre-steps are needed. 

```
x, y = clean_data(ds)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=42)

automl_training_set = pd.concat([x_train,y_train],axis=1)

```

The data has to be cleaned and split in a train and test set. For this, the clean_data and train_test_split methods of the previous sklearn pipeline are used. Finally, the x_train and y_train dataframes have to be combined for the AutoML config.


To execute a run an [AutoMLConfig](https://docs.microsoft.com/en-us/python/api/azureml-train-automl-client/azureml.train.automl.automlconfig.automlconfig?view=azure-ml-py) object is needed:

```
automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task="classification",
    training_data=automl_training_set,    
    iterations=30,
    iteration_timeout_minutes=5,
    primary_metric="accuracy",
    label_column_name='y',
    n_cross_validations=5)

```
With this configuration the AutoML task was executed.

The best solution was a VotingEnsemble model. This algorithm combines several different algorithms and takes the majority of the votes. The result is a very robust model. These are the chosen alogrithms and their specific weights:


```

"ensembled_algorithms": "['LightGBM', 'XGBoostClassifier', 'LightGBM', 'SGD', 'LightGBM', 'SGD', 'RandomForest', 'ExtremeRandomTrees']",

"ensemble_weights": "[0.23076923076923078, 0.3076923076923077, 0.07692307692307693, 0.07692307692307693, 0.07692307692307693, 0.07692307692307693, 0.07692307692307693, 0.07692307692307693]"


```

With the Azure SDK it's possible to get deeper insights in the VotingEmsemble model. This is an extract of the file 'VotingEnsemble_explained.txt' that visualizes details of the XGBoostClassifier (especially the hyperparameters) in the ensemble:

```

xgboostclassifier
{'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 1,
 'colsample_bynode': 1,
 'colsample_bytree': 1,
 'gamma': 0,
 'learning_rate': 0.1,
 'max_delta_step': 0,
 'max_depth': 3,
 'min_child_weight': 1,
 'missing': nan,
 'n_estimators': 100,
 'n_jobs': 1,
 'nthread': None,
 'objective': 'binary:logistic',
 'random_state': 0,
 'reg_alpha': 0,
 'reg_lambda': 1,
 'scale_pos_weight': 1,
 'seed': None,
 'silent': None,
 'subsample': 1,
 'tree_method': 'auto',
 'verbose': -10,
 'verbosity': 0}


```


## Pipeline comparison

__1. LogisticRegression results__
   
For finding the best hyperparameters Hyperdrive was executed 50 times. Attached you can see the ranking of the best runs:

![Best Hyperdrive runs](/final_pictures/LR_runs.PNG)

It's visible that the run results don't differ a lot. The best run has accuracy score of 0,91085, that's really good. Significant is the fact that the "best" runs were all executed 1000 iterations. Small changes in the C parameter seem to affect the accuracy only a bit. The next picture visualizes the accuracy scores of the different runs in one chart.

![Accuracy metric](/final_pictures/LR_accuracy.PNG)

It's very clear that runs with specific parameter sets are all in the same range. There are accumulations around 0.91, 0.904 and 0.9. The best model has the hyperparameters C = 0.80077 and max_iter = 1000.


__2. AutoML results__
   
The AutoML routine was executed 30 times. Attached to can see the best runs:

![Best AutoML runs](/final_pictures/AutoML_run.PNG)

Similar to the different LogisticRegression runs the accuracy of the best models also differ only a bit. It's very characteristic that the first best runs are both Ensemble algorithms. The combination of different estimators achieves great performance and robustness.  

__3. Comparison__
 
In the end, there is nearly not a difference between the best AutoML model (0.91720) and the LogisticRegression model with optimized hyperparameters (0.91085). To get a better feeling of the data, AutoML can visualize the features that have the greatest influence on the model.


![Global importance](/final_pictures/Global_importance.PNG)


The four most importance features are duration, nr.employes, cons.conf.idx and emp.var.rate. With this knowledge, it's very interesting to explore the distribution of values in these feature categories.


![Summary importance](/final_pictures/Summary_importance.PNG)


It's very clear to see that the distribution is highly imbalanced for most of the features. An imbalanced dataset can affect the model performance negativly. The model cannot learn from data that is not available in the dataset. If such a value is used in inference mode the model can only guess and this results in a worse accurcay. 

Nevertheless both models seem to be robust against imbalanced data because the accuracy is very good. A good additional metric is the convolution matrix to get a feeling how the model (in this case the VotingEnsemble) performs in detail.

![Metric](/final_pictures/metric_conv.PNG)

It's very clear that the model has a great performance in the prediction of the class 0 with 0.9612. On ther other hand the prediction of label 1 is not really good with 0.5711. This is again an indicatior for a highly imbalanced dataset.


## Future work
Different points could be improved.

1. __sklearn pipeline__
   
The initial results of the model were really good. Nevertheless some test with the different policies and parameter sampler could be helpful. The combination of a Median stopping policy and a Bayesian sampling without parallel node computation could lead to better results. 

2. __AutoML__
   
The first results of AutoML were also really good. It could be helpful to run more iterations and models to achieve better results. Checking out the best runs on the picture above shows that the first models are Ensemble approaches and nearly all are in the context of boosting. This is an indicator that boosting and especially the combination of different boosting algorithms in an Ensemble is a good approach. An option that could lead to better results is the use of deep learning in the AutoML config. Maybe a neural network can map the complex internal relationships to a robust model.


3. __General__
   
In my opinion the most powerful but also most time-consuming approach is the update of the dataset. A highly unbalanced dataset is not a good base for a robust machine learning algorithm. Maybe it is possible to update the set with missing data to achieve a balanced feature distribution. If this is not possible, resampling algorithms could be used to provide the model with training data that is equally distributed.



