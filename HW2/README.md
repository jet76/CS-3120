# Homework 2 - Logistic Regression for binary classification  
(NumPy, Pandas and data visualization packages are allowed.)  
(SKLearn regression models are allowed!)  
Reference code: 2_Logostic_ExSKLearn_Demo.py in blackboard

1.	**Select a dataset with binary target values** using 
https://machinelearningmastery.com/standard-machine-learning-datasets/
e.g. banknote or diabetes dataset

2.	**Use pandas to read CSV file as dataframe.** (1pt)
**e.g. The following code helps import pima diabetes dataset**
```
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
pima = pd.read_csv("pima-indians-diabetes-database.csv", header=None, names=col_names)
```

3.	**Select 5 (if not possible then select 4) features from the chosen dataset.**  (1pt) 
**List all features you selected in your report.**
For example, the following code will select two features 
```
feature_cols = ['pregnant', 'age']
X = pima[feature_cols]
```

4.	Use “train _test_split” from “sklearn.cross_validationtrain” to split test and training data by 40% testing + 60% training.   (1pt)

5.	Fit your model with training data and test your model after fitting. 

6.	Calculate and plot out  
the confusion matrix  (1pt)  
precision score, recall score, F score (3pts)  
**Copy your console output (these scores) to your report.**

7.	Plot out the ROC curve and print out the ROC_AUC score (sklearn.metrics.roc_curve() and sklearn.metrics.roc_auc_score() can be used.) (3pts)
