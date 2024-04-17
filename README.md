# Home Loan Project

## Task 2

Standard Bank is embracing the digital transformation wave and intends
to use new and exciting technologies to give their customers a complete
set of services from the convenience of their mobile devices. As
Africa's biggest lender by assets, the bank aims to improve the current
process in which potential borrowers apply for a home loan. The current
process involves loan officers having to manually process home loan
applications. This process takes 2 to 3 days to process upon which the
applicant will receive communication on whether or not they have been
granted the loan for the requested amount. To improve the process
Standard Bank wants to make use of machine learning to assess the credit
worthiness of an applicant by implementing a model that will predict if
the potential borrower will default on his/her loan or not, and do this
such that the applicant receives a response immediately after completing
their application.

You will be required to follow the data science lifecycle to fulfill the
objective. The data science lifecycle
(<https://www.datascience-pm.com/crisp-dm-2/>) includes:

-   Business Understanding
-   Data Understanding
-   Data Preparation
-   Modelling
-   Evaluation
-   Deployment.

You now know the CRoss Industry Standard Process for Data Mining
(CRISP-DM), have an idea of the business needs and objectivess, and
understand the data. Next is the tedious task of preparing the data for
modeling, modeling and evaluating the model. Luckily, just like EDA the
first of the two phases can be automated. But also, just like EDA this
is not always best.

In this task you will be get a taste of AutoML and Bespoke ML. In the
notebook we make use of the library auto-sklearn/autosklearn
(<https://www.automl.org/automl/auto-sklearn/>) for AutoML and sklearn
for ML. We will use train one machine for the traditional approach and
you will be required to change this model to any of the models that
exist in sklearn. The model we will train will be a Logistic Regression.
Parts of the data preparation will be omitted for you to do, but we will
provide hints to lead you in the right direction.

The data provided can be found in the Resources folder as well as
(<https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset>).

-   train will serve as the historical dataset that the model will be
    trained on and,
-   test will serve as unseen data we will predict on, i.e. new
    (\'future\') applicants.

### Part One

There are many AutoEDA Python libraries out there which include:

-   dtale (<https://dtale.readthedocs.io/en/latest/>)
-   pandas profiling
    (<https://pandas-profiling.ydata.ai/docs/master/index.html>)
-   autoviz (<https://readthedocs.org/projects/autoviz/>)
-   sweetviz (<https://pypi.org/project/sweetviz/>)

and many more. In this task we will use Sweetviz.. You may be required
to use bespoke EDA methods.

The Home Loans Department manager wants to know the following:

1.  An overview of the data. (HINT: Provide the number of records,
    fields and their data types. Do for both).

2.  What data quality issues exist in both train and test? (HINT:
    Comment any missing values and duplicates)

3.  How do the the loan statuses compare? i.e. what is the distrubition
    of each?

4.  How do women and men compare when it comes to defaulting on loans in
    the historical dataset?

5.  How many of the loan applicants have dependents based on the
    historical dataset?

6.  How do the incomes of those who are employed compare to those who
    are self employed based on the historical dataset?

7.  Are applicants with a credit history more likely to default than
    those who do not have one?

8.  Is there a correlation between the applicant\'s income and the loan
    amount they applied for?

### Part Two

Run the AutoML section and then fill in code for the traditional ML
section for the the omitted cells.

Please note that the notebook you submit must include the analysis you
did in Task 2.
## Import Libraries
``` python
!pip install sweetviz 
#uncomment the above if you need to install the library 
```



``` python
!pip install --upgrade scipy
```

``` python
!pip install autosklearn
```

``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
```

``` python
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
```

# Part One

``` python
train_data.head()
```

``` python
test_data.head()
```


``` python
# we concat for easy analysis
n = train_data.shape[0] # we set this to be able to separate the
df = pd.concat([train_data, test_data], axis=0)
df.head()
```

``` python
print(df)
```


### Sweetviz
``` python
autoEDA = sweetviz.analyze(train_data)
autoEDA.show_notebook()
```


### Your Own EDA

Question 1. An overview of the data. (Provide the number of records,
fields and their data types. Do for both).
``` python
## overview of train dataset
train_data.info()
```

``` python
## overview of test dataset
test_data.info()
```

``` python
## Statistical view of train dataset
train_data.describe(include='all').T
```


Question 2. What data quality issues exist in both train and test?
(Comment any missing values and duplicates)
``` python
## checking duplicate values of train dataset
train_data.duplicated().sum()
```

``` python
## checking duplicate values of test dataset
test_data.duplicated().sum()
```

``` python
## checking missing values of train dataset
train_data.isnull().sum()
```

``` python
## checking missing values of test dataset
test_data.isnull().sum()
```

``` python
## Visualizing Missing Data of train dataset
sns.heatmap(train_data.isnull())
plt.show()
```
![12dccf35a6d14faf193a1b5731ff5fdf73c1bc3f](https://github.com/munaf101/Home-Loan-Project/assets/105988451/c4f5ec63-5014-426b-96fa-57be70d49a2f)

``` python
## Visualizing Missing Data of test dataset
sns.heatmap(test_data.isnull())
plt.show()
```
![463791f8b5828de1f1bd4e9f2fb0e7d1dc493cf1](https://github.com/munaf101/Home-Loan-Project/assets/105988451/ff96a9c1-2854-4cea-90e3-28f2e5cfba9d)

``` python
print("\n ----- Value Counts ----- \n")
print(train_data["Loan_Status"].value_counts())

print("\n ------ Normalize Value ------ \n")
train_data["Loan_Status"].value_counts(normalize=True)
```

``` python
value_counts = [422, 192]
normalize_value = [0.687296, 0.312704]
plt.figure(figsize=(6,4))
sns.barplot(x=["Y", "N"], y=value_counts, palette={"Y": "blue", "N": "red"})
plt.title("Distrubition of Loan Status", fontsize=20, fontweight='bold', color='brown')
plt.show()

```
![6e582e00497036f1c27c2b26cb4a1d9f5bc0b866](https://github.com/munaf101/Home-Loan-Project/assets/105988451/b2a87c1b-d9da-488e-9cf4-4083804d7d52)

Question 4. How do women and men compare when it comes to defaulting on
loans in the historical dataset?
``` python
print("\n ----- Value Counts ----- \n")
print(train_data["Gender"].value_counts())

print("\n ----- Gender wise Loan Status ----- \n")
train_data.groupby("Gender")["Loan_Status"].value_counts()
```

``` python
value_counts = [422, 192, 489, 112]
plt.figure(figsize=(6,4))
sns.barplot(x=["Y (Male)", "Y (Female)", "N (Male)", "N (Female)"], y=value_counts, palette={"Y (Male)": "blue", "Y (Female)": "blue", "N (Male)": "red", "N (Female)": "red"})
plt.title("Gender-wise Loan Status Comparison", fontsize=20, fontweight='bold', color='brown')
plt.show()
```
![46f95ee10555dbfd8e8a62e2733488f61834d0d1](https://github.com/munaf101/Home-Loan-Project/assets/105988451/501dc66e-0ff1-431a-b38f-a13d33b63e66)

Question 5.How many of the loan applicants have dependents based on the
historical dataset?
``` python
train_data[train_data["Dependents"] != "0"].shape[0]
```

``` python
train_data[train_data["Dependents"] != '0'].shape[0]/train_data.shape[0]
```

Question 6. How do the incomes of those who are employed compare to
those who are self employed based on the historical dataset?
``` python
train_data.groupby("Self_Employed")["ApplicantIncome"].describe()
```


Question 7. Are applicants with a credit history more likely to default
than those who do not have one?
``` python
plt.figure(figsize=(6,4))
sns.countplot(x='Credit_History', data=train_data)
plt.title("Distrubition of Credit History based on Loan Status", fontsize=15, fontweight='bold', color='red')
plt.show()
```
![c61018c25cef8d1b1d6733d932757de8e9cefd5c](https://github.com/munaf101/Home-Loan-Project/assets/105988451/77e1c14b-a9d1-483c-99a5-c0ec6ec7afe0)


print("\n ----- Value Counts ----- \n")
print(train_data["Credit_History"].value_counts())

print("\n ---------------------------- \n")
train_data.groupby('Credit_History')['Loan_Status'].value_counts()
```
``` python
value_counts = [475, 89]

plt.figure(figsize=(6,4))
sns.barplot(x=["Y", "N"], y=value_counts, palette={"Y": "blue", "N": "red"})
plt.title("Distrubition of Credit History based on Loan Status", fontsize=20, fontweight='bold', color='brown')
plt.show()
```

Question 8.Is there a correlation between the applicant\'s income and
the loan amount they applied for?
``` python
train_data.corr()
```

``` python
# Correlation Heatmap
plt.figure(figsize=(12,7))
sns.heatmap(train_data.corr(), annot=True, cmap='RdYlGn', vmax=.5)
plt.show()
```
![af69f06d5f71135cd970c889fbcef10529a59ad0](https://github.com/munaf101/Home-Loan-Project/assets/105988451/2da25499-b5d7-40d1-9ef7-a7dd96b2b201)


Answers:- Q1 - An overview of the data.

Train Data contains 614 Rows and total 13 columns.

Out of 13 columns there are 4 float columns, 1 integer columns and 8
object columns.

Test Data contains 367 Rows and total 12 columns.

And out of 12 columns there are 3 float columns, 2 Integer columns and 7
object columns.

Q2 - What data quality issues exist in both train and test?

There are no duplicate values in both train dataset and test dataset.

But both dataset have some missing values.

Train Dataset: Gender 13, Married 3, Dependents 15, Self_Employed 32,
LoanAmount 22, Loan_Amount_Term 14 and Credit_History 50 contains
missing values respectively.

Test Dataset: Gender 11, Dependents 10, Self_Employed 23, LoanAmount 5,
Loan_Amount_Term 6 and Credit_History 29 contains missing values
respectively.

Q3 - How do the loan statuses compare? i.e. what is the distrubition of
each?

There are 422 loans with a Yes status, which is the majority, and 192
loans with No status.

Q4 - How do women and men compare when it comes to defaulting on loans
in the historical dataset?

Men have a higher loan status of Yes than women do, and same goes to
loan status of No.

Q5 - How many of the loan applicants have dependents based on the
historical dataset?

According to the historical dataset, 269 loan applicants have
dependents.

Q6 - How does the income of those who are employed compare to those who
are self-employed based on the historical dataset?

The average income of those who are employed is 5049 which is low
compared to self-employed that is 7380. The minimum income of those who
are employed is 150 which is low compared to self-employed that is 674.
But the maximum income of those who are employed is 81000 which is high
compared to self-employed that is 39147. Additionally, 500 people are
employed, which is a large number when compared to the 82 people who
were self-employed in the historical dataset.

Q7 - Are applicants with a credit history more likely to default than
those who do not have one?

Indeed, candidates with credit histories may be more likely to default
than those without them.

Q8 - Is there a correlation between the applicant\'s income and the loan
amount they applied for?

Yes, there is a correlation between the applicant\'s income and the loan
amount they applied for.
# Part Two
## Auto ML wth autosklearn
``` python
pip install autosklearn
```

``` python
import autosklearn.classification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
```
``` python
pip install --upgrade scikit-learn
```

``` python
### Matrix of features

X = train_data[['Gender',
'Married',
'Dependents',
'Education',
'Self_Employed',
'ApplicantIncome',
'CoapplicantIncome',
'LoanAmount',
'Loan_Amount_Term',
'Credit_History',
'Property_Area']]

### convert string(text) to categorical
X['Gender'] = X['Gender'].astype('category')
X['Married'] = X['Married'].astype('category')
X['Education'] = X['Education'].astype('category')
X['Dependents'] = X['Dependents'].astype('category')
X['Self_Employed'] = X['Self_Employed'].astype('category')
X['Property_Area'] = X['Property_Area'].astype('category')


### label encode target
y = train_data['Loan_Status'].map({'N':0,'Y':1}).astype(int)


### train-test split
#from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


``` python
#train
autoML = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=2*30, per_run_time_limit=30, n_jobs=8)
autoML.fit(X_train, y_train)

# predict
predictions_autoML = autoML.predict(X_test)
```
``` python
print('Model Accuracy:', accuracy_score(predictions_autoML, y_test))
```

``` python
print(confusion_matrix(predictions_autoML, y_test))
```

``` python
print(classification_report(predictions_autoML, y_test))
```

``` python
con_mat = confusion_matrix(predictions_autoML, y_test)

sns.heatmap(con_mat, annot=True)
plt.show()
```
![31a1ac05672be676dcb9a239bc6fdcdc663b610b](https://github.com/munaf101/Home-Loan-Project/assets/105988451/a8fc7e07-e395-4cb7-8cce-f112d6157a2c)

## Bespoke ML sklearn

### Data Preparation
``` python
# Matrix of features

df = train_data[['Education',
'Property_Area']]

### Include Numerical Features Here ###
### Handle Missing Values Here ###
### Scale Here ###


# label encode target
y = train_data['Loan_Status'].map({'N':0,'Y':1}).astype(int)

# # encode with get dummies
X = pd.DataFrame(df, columns=df.columns)
X = pd.get_dummies(X, drop_first=True)

# # train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
``` python
import sklearn
import scipy
```
``` python
# some classifiers you can pick from (remember to import)

classifiers = sklearn.utils.all_estimators(type_filter=None)
for name, class_ in classifiers:
    if hasattr(class_, 'predict_proba'):
        print(name)
```

``` python
# train
clf = LogisticRegression() #change model here
clf.fit(X_train, y_train)

# predict
predictions_clf = clf.predict(X_test)
```
``` python
print('Model Accuracy:', accuracy_score(predictions_clf, y_test))
```

``` python
print(confusion_matrix(predictions_clf, y_test))
```

``` python
import warnings
warnings.filterwarnings('ignore')
```
``` python
# Matrix of features

df = train_data[['Gender',
'Married',
'Education',
'Self_Employed',
'ApplicantIncome',
'CoapplicantIncome',
'LoanAmount',
'Loan_Amount_Term',
'Credit_History']]


# imputing the missing values:
df['Gender'].fillna(df['Gender'].mode()[0], inplace = True)
df['Married'].fillna(df['Married'].mode()[0], inplace = True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace = True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace = True)

# encoding categorical features
df['Gender'] = df['Gender'].map({'Male':0,'Female':1}).astype(int)
df['Married'] = df['Married'].map({'No':0,'Yes':1}).astype(int)
df['Education'] = df['Education'].map({'Not Graduate':0,'Graduate':1}).astype(int)
df['Self_Employed'] = df['Self_Employed'].map({'No':0,'Yes':1}).astype(int)
df['Credit_History'] = df['Credit_History'].astype(int)


df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace = True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(), inplace = True)
 
X = df.copy()

# label encode target
y = train_data['Loan_Status'].map({'N':0,'Y':1}).astype(int)


# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
``` python
# train
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier() #change model here
clf.fit(X_train, y_train)

# predict
predictions_clf = clf.predict(X_test)
```
``` python
print('Model Accuracy:', accuracy_score(predictions_clf, y_test))
```

``` python
```
``` python
print(confusion_matrix(predictions_clf, y_test))
```

``` python
cm = confusion_matrix(predictions_clf, y_test)
sns.heatmap(cm, annot=True)
plt.show()
```
![4d6f1db3082eed4fc80f4137f414d38cb558a7f4](https://github.com/munaf101/Home-Loan-Project/assets/105988451/4be6a55d-109e-4593-8d0e-ee27ecdd2928)
## ROC - AUC
``` python
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(predictions_clf, y_test)
print("AUC score is ", auc)
```

``` python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(predictions_clf, y_test)

plt.figure(figsize=(8, 6))

plt.plot(fpr, tpr, color="darkgreen", label="ROC")
plt.plot([0, 1], [0, 1], color="orange", linestyle="--", label="ROC Curve (area = %0.2f)" % auc)

plt.xlabel("False Positive Rate", fontsize=15, fontweight="bold", color="brown")
plt.ylabel("True Positive Rate", fontsize=15, fontweight="bold", color="brown")
plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=20, fontweight="bold", color="brown")
plt.legend()

plt.show()
```
![59a8132b565a3f34acbe965d9c6718cd91c98d5c](https://github.com/munaf101/Home-Loan-Project/assets/105988451/a4e78500-c369-4141-8895-fcd729173bd2)

``` python
# Applying 5-Fold Cross Validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, X_train, y_train, cv = 5, scoring = "accuracy")

print("Cross-Validation score:{}".format(scores))
```

``` python
# compute Average cross-validation score
#the cross-validation accuracy by calculating its mean
print("Average cross-validation score: {:.4f}".format(scores.mean()))
```

