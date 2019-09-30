
#### MACHINE LEARNING EXERCISE: CLASSIFICATION
# TELCO CUSTOMER CHURN

#### Models
* Logistic Regression
* K-Nearest Neighbors
* Naive Bayes
* Decision Trees
* Random Forest Classifier
* XGBoost

#### About
* "Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs." [IBM Sample Data Sets]

#### Target Variable
* Churn (Yes / No)

#### Features
1. gender
1. SeniorCitizen
1. Partner
1. Dependents
1. tenure
1. PhoneService
1. MultipleLines
1. InternetService
1. OnlineSecurity
1. OnlineBackup
1. DeviceProtection
1. TechSupport
1. StreamingTV
1. StreamingMovies
1. Contract
1. PaperlessBilling
1. PaymentMethod
1. MonthlyCharges
1. TotalCharges

#### Source
* https://www.kaggle.com/blastchar/telco-customer-churn

## Import Libraries


```python
##### Standard Libraries #####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("poster")

%matplotlib inline
```


```python
##### Other Libraries #####

## ML Algorithms ##
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

## To visualize decision tree ##
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

## For building models ##
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from sklearn.pipeline import make_pipeline

## For measuring performance ##
from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_validate 

## Ignore warnings ##
import warnings
warnings.filterwarnings('ignore')
```

    

## Load the Dataset
First of all, let us load the dataset then check if it is properly loaded by showing a snippet of the data and checking its columns.


```python
### Load the data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv", index_col="customerID")

### Check if the data is properly loaded
print("Size of the dataset:", df.shape)
df.head()
```

    Size of the dataset: (7043, 20)
    





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>OnlineBackup</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
    <tr>
      <th>customerID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7590-VHVEG</th>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>1</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>No</td>
    </tr>
    <tr>
      <th>5575-GNVDE</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>34</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>56.95</td>
      <td>1889.5</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3668-QPYBK</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>7795-CFOCW</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>45</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1840.75</td>
      <td>No</td>
    </tr>
    <tr>
      <th>9237-HQITU</th>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>



From the data snippet above, majority of the columns have data in words/strings. Only `tenure`, `MonthlyCharges` and `TotalCharges` are the columns containing continuous numbers. While the `SeniorCitizen` also contains numbers, it only has two distinct values which are `0` and `1`.

Below is the list of columns of this dataset, along with the datatypes of the columns.


```python
### List the columns along with its type
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 7043 entries, 7590-VHVEG to 3186-AJIEK
    Data columns (total 20 columns):
    gender              7043 non-null object
    SeniorCitizen       7043 non-null int64
    Partner             7043 non-null object
    Dependents          7043 non-null object
    tenure              7043 non-null int64
    PhoneService        7043 non-null object
    MultipleLines       7043 non-null object
    InternetService     7043 non-null object
    OnlineSecurity      7043 non-null object
    OnlineBackup        7043 non-null object
    DeviceProtection    7043 non-null object
    TechSupport         7043 non-null object
    StreamingTV         7043 non-null object
    StreamingMovies     7043 non-null object
    Contract            7043 non-null object
    PaperlessBilling    7043 non-null object
    PaymentMethod       7043 non-null object
    MonthlyCharges      7043 non-null float64
    TotalCharges        7043 non-null object
    Churn               7043 non-null object
    dtypes: float64(1), int64(2), object(17)
    memory usage: 660.3+ KB
    

Columns with data in *Strings* are considered *object* while columns with numerical data are either *int* or *float*.

The only numerical columns should be `SeniorCitizen`, `tenure`, `MonthlyCharges` and `TotalCharges`. But the column type of`TotalCharges` is object, so we shall inspect it further.

## Explore the Dataset

Let us look at the summary of statistics of the data. For now, the numerical columns will only be displayed.


```python
### Summary of statistics for numerical data
df.describe()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SeniorCitizen</th>
      <th>tenure</th>
      <th>MonthlyCharges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7043.000000</td>
      <td>7043.000000</td>
      <td>7043.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.162147</td>
      <td>32.371149</td>
      <td>64.761692</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.368612</td>
      <td>24.559481</td>
      <td>30.090047</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>18.250000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>35.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>29.000000</td>
      <td>70.350000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>55.000000</td>
      <td>89.850000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>72.000000</td>
      <td>118.750000</td>
    </tr>
  </tbody>
</table>
</div>



Notice that the minimum value at `tenure` column is 0, which may mean that that customer just started availing the service.

Now, let us look at the summary of categorical columns.


```python
### Summary of statistics for categorical data
df.describe(include="O")
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>OnlineBackup</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7043</td>
      <td>7043</td>
      <td>7043</td>
      <td>7043</td>
      <td>7043</td>
      <td>7043</td>
      <td>7043</td>
      <td>7043</td>
      <td>7043</td>
      <td>7043</td>
      <td>7043</td>
      <td>7043</td>
      <td>7043</td>
      <td>7043</td>
      <td>7043</td>
      <td>7043</td>
      <td>7043</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>6531</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Male</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td></td>
      <td>No</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>3555</td>
      <td>3641</td>
      <td>4933</td>
      <td>6361</td>
      <td>3390</td>
      <td>3096</td>
      <td>3498</td>
      <td>3088</td>
      <td>3095</td>
      <td>3473</td>
      <td>2810</td>
      <td>2785</td>
      <td>3875</td>
      <td>4171</td>
      <td>2365</td>
      <td>11</td>
      <td>5174</td>
    </tr>
  </tbody>
</table>
</div>



From the table above, we can see the number of distinct values and the most common value per categorical column. 

If we look at `TotalCharges` column, the top values is `" "` or space. It has 6531 distinct values out of 7043 rows which is very unideal for a categorical column.

Let us list the distinct values of `TotalCharges` column.


```python
### Find the string in TotalCharges column
df["TotalCharges"].value_counts()[:10]
```




             11
    20.2     11
    19.75     9
    19.9      8
    19.65     8
    20.05     8
    45.3      7
    19.55     7
    20.15     6
    19.45     6
    Name: TotalCharges, dtype: int64



Looks like the `" "`/space values in the `TotalCharges` column cause problems in parsing this column into numerical.

Below are the 11 rows containing `TotalCharges` equals to space.


```python
### Inspect the rows with TotalCharges==" "
df[ df["TotalCharges"]==" " ]
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>OnlineBackup</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
    <tr>
      <th>customerID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4472-LVYGI</th>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Bank transfer (automatic)</td>
      <td>52.55</td>
      <td></td>
      <td>No</td>
    </tr>
    <tr>
      <th>3115-CZMZD</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>Yes</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>Two year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>20.25</td>
      <td></td>
      <td>No</td>
    </tr>
    <tr>
      <th>5709-LVOEQ</th>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Two year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>80.85</td>
      <td></td>
      <td>No</td>
    </tr>
    <tr>
      <th>4367-NUYAO</th>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>Two year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>25.75</td>
      <td></td>
      <td>No</td>
    </tr>
    <tr>
      <th>1371-DWPAZ</th>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Two year</td>
      <td>No</td>
      <td>Credit card (automatic)</td>
      <td>56.05</td>
      <td></td>
      <td>No</td>
    </tr>
    <tr>
      <th>7644-OMVMY</th>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>Two year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>19.85</td>
      <td></td>
      <td>No</td>
    </tr>
    <tr>
      <th>3213-VVOLG</th>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>Two year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>25.35</td>
      <td></td>
      <td>No</td>
    </tr>
    <tr>
      <th>2520-SGTTA</th>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>Two year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>20.00</td>
      <td></td>
      <td>No</td>
    </tr>
    <tr>
      <th>2923-ARZLG</th>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>One year</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>19.70</td>
      <td></td>
      <td>No</td>
    </tr>
    <tr>
      <th>4075-WKNIU</th>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>DSL</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Two year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>73.35</td>
      <td></td>
      <td>No</td>
    </tr>
    <tr>
      <th>2775-SEFEE</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>Yes</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Bank transfer (automatic)</td>
      <td>61.90</td>
      <td></td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



We can observed from the table above that the rows where `TotalCharges == " "` did not churn, but most notably, also have `tenure == 0`. So, let's inpect further rows wherein `tenure == 0`.


```python
### Display the number of zero values 
print("---Count zero values---")
print("Tenure: {}".format( df["tenure"].value_counts()[0] ))

df[ df["tenure"]==0 ]
```

    ---Count zero values---
    Tenure: 11
    





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>OnlineBackup</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
    <tr>
      <th>customerID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4472-LVYGI</th>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Bank transfer (automatic)</td>
      <td>52.55</td>
      <td></td>
      <td>No</td>
    </tr>
    <tr>
      <th>3115-CZMZD</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>Yes</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>Two year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>20.25</td>
      <td></td>
      <td>No</td>
    </tr>
    <tr>
      <th>5709-LVOEQ</th>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Two year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>80.85</td>
      <td></td>
      <td>No</td>
    </tr>
    <tr>
      <th>4367-NUYAO</th>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>Two year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>25.75</td>
      <td></td>
      <td>No</td>
    </tr>
    <tr>
      <th>1371-DWPAZ</th>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Two year</td>
      <td>No</td>
      <td>Credit card (automatic)</td>
      <td>56.05</td>
      <td></td>
      <td>No</td>
    </tr>
    <tr>
      <th>7644-OMVMY</th>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>Two year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>19.85</td>
      <td></td>
      <td>No</td>
    </tr>
    <tr>
      <th>3213-VVOLG</th>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>Two year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>25.35</td>
      <td></td>
      <td>No</td>
    </tr>
    <tr>
      <th>2520-SGTTA</th>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>Two year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>20.00</td>
      <td></td>
      <td>No</td>
    </tr>
    <tr>
      <th>2923-ARZLG</th>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>One year</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>19.70</td>
      <td></td>
      <td>No</td>
    </tr>
    <tr>
      <th>4075-WKNIU</th>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>DSL</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Two year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>73.35</td>
      <td></td>
      <td>No</td>
    </tr>
    <tr>
      <th>2775-SEFEE</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>Yes</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Bank transfer (automatic)</td>
      <td>61.90</td>
      <td></td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



This proves that the rows that have `TotalCharges == " "` also have `tenure == 0`. This means that these customers just availed the service and have not been regularly charged for the service. 

Let us remove these rows since it has incomplete data and may cause problems for modelling.


```python
### Remove rows where tenure = 0 and TotalCharges is not numerical 
df1 = df.drop(df[ df["tenure"]==0 ].index, axis=0)

### Convert TotalCharges column from String to float
df1["TotalCharges"] = df1["TotalCharges"].astype(float)

### Check df1 to see if transformations are successful
df1.describe()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SeniorCitizen</th>
      <th>tenure</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7032.000000</td>
      <td>7032.000000</td>
      <td>7032.000000</td>
      <td>7032.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.162400</td>
      <td>32.421786</td>
      <td>64.798208</td>
      <td>2283.300441</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.368844</td>
      <td>24.545260</td>
      <td>30.085974</td>
      <td>2266.771362</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>18.250000</td>
      <td>18.800000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>35.587500</td>
      <td>401.450000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>29.000000</td>
      <td>70.350000</td>
      <td>1397.475000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>55.000000</td>
      <td>89.862500</td>
      <td>3794.737500</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>72.000000</td>
      <td>118.750000</td>
      <td>8684.800000</td>
    </tr>
  </tbody>
</table>
</div>



Now that the problems with `tenure` and `TotalCharges` are fixed, we can further clean the data to be machine readable. 

Let's convert those categories or strings per column into numerical labels using `LabelEncoder`. Listed below are the categories per column along with its transformation into numerical labels.


```python
df_clean = df1
le = {}

print("-----Value for Categorical Columns-----")
for col in df_clean.columns:
    if not col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        print( "--{}--\n{}".format(col, df_clean[col].value_counts()) )
        le[col] = LabelEncoder()
        df_clean[col] = le[col].fit_transform(df_clean[col])
        
        print( "Transformed:\n{}\n".format(df_clean[col].value_counts()) )
```

    -----Value for Categorical Columns-----
    --gender--
    Male      3549
    Female    3483
    Name: gender, dtype: int64
    Transformed:
    1    3549
    0    3483
    Name: gender, dtype: int64
    
    --Partner--
    No     3639
    Yes    3393
    Name: Partner, dtype: int64
    Transformed:
    0    3639
    1    3393
    Name: Partner, dtype: int64
    
    --Dependents--
    No     4933
    Yes    2099
    Name: Dependents, dtype: int64
    Transformed:
    0    4933
    1    2099
    Name: Dependents, dtype: int64
    
    --PhoneService--
    Yes    6352
    No      680
    Name: PhoneService, dtype: int64
    Transformed:
    1    6352
    0     680
    Name: PhoneService, dtype: int64
    
    --MultipleLines--
    No                  3385
    Yes                 2967
    No phone service     680
    Name: MultipleLines, dtype: int64
    Transformed:
    0    3385
    2    2967
    1     680
    Name: MultipleLines, dtype: int64
    
    --InternetService--
    Fiber optic    3096
    DSL            2416
    No             1520
    Name: InternetService, dtype: int64
    Transformed:
    1    3096
    0    2416
    2    1520
    Name: InternetService, dtype: int64
    
    --OnlineSecurity--
    No                     3497
    Yes                    2015
    No internet service    1520
    Name: OnlineSecurity, dtype: int64
    Transformed:
    0    3497
    2    2015
    1    1520
    Name: OnlineSecurity, dtype: int64
    
    --OnlineBackup--
    No                     3087
    Yes                    2425
    No internet service    1520
    Name: OnlineBackup, dtype: int64
    Transformed:
    0    3087
    2    2425
    1    1520
    Name: OnlineBackup, dtype: int64
    
    --DeviceProtection--
    No                     3094
    Yes                    2418
    No internet service    1520
    Name: DeviceProtection, dtype: int64
    Transformed:
    0    3094
    2    2418
    1    1520
    Name: DeviceProtection, dtype: int64
    
    --TechSupport--
    No                     3472
    Yes                    2040
    No internet service    1520
    Name: TechSupport, dtype: int64
    Transformed:
    0    3472
    2    2040
    1    1520
    Name: TechSupport, dtype: int64
    
    --StreamingTV--
    No                     2809
    Yes                    2703
    No internet service    1520
    Name: StreamingTV, dtype: int64
    Transformed:
    0    2809
    2    2703
    1    1520
    Name: StreamingTV, dtype: int64
    
    --StreamingMovies--
    No                     2781
    Yes                    2731
    No internet service    1520
    Name: StreamingMovies, dtype: int64
    Transformed:
    0    2781
    2    2731
    1    1520
    Name: StreamingMovies, dtype: int64
    
    --Contract--
    Month-to-month    3875
    Two year          1685
    One year          1472
    Name: Contract, dtype: int64
    Transformed:
    0    3875
    2    1685
    1    1472
    Name: Contract, dtype: int64
    
    --PaperlessBilling--
    Yes    4168
    No     2864
    Name: PaperlessBilling, dtype: int64
    Transformed:
    1    4168
    0    2864
    Name: PaperlessBilling, dtype: int64
    
    --PaymentMethod--
    Electronic check             2365
    Mailed check                 1604
    Bank transfer (automatic)    1542
    Credit card (automatic)      1521
    Name: PaymentMethod, dtype: int64
    Transformed:
    2    2365
    3    1604
    0    1542
    1    1521
    Name: PaymentMethod, dtype: int64
    
    --Churn--
    No     5163
    Yes    1869
    Name: Churn, dtype: int64
    Transformed:
    0    5163
    1    1869
    Name: Churn, dtype: int64
    
    


```python
df_clean.describe()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>OnlineBackup</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7032.000000</td>
      <td>7032.000000</td>
      <td>7032.000000</td>
      <td>7032.000000</td>
      <td>7032.000000</td>
      <td>7032.000000</td>
      <td>7032.000000</td>
      <td>7032.000000</td>
      <td>7032.000000</td>
      <td>7032.000000</td>
      <td>7032.000000</td>
      <td>7032.000000</td>
      <td>7032.000000</td>
      <td>7032.000000</td>
      <td>7032.000000</td>
      <td>7032.000000</td>
      <td>7032.000000</td>
      <td>7032.000000</td>
      <td>7032.000000</td>
      <td>7032.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.504693</td>
      <td>0.162400</td>
      <td>0.482509</td>
      <td>0.298493</td>
      <td>32.421786</td>
      <td>0.903299</td>
      <td>0.940557</td>
      <td>0.872582</td>
      <td>0.789249</td>
      <td>0.905859</td>
      <td>0.903868</td>
      <td>0.796359</td>
      <td>0.984926</td>
      <td>0.992890</td>
      <td>0.688567</td>
      <td>0.592719</td>
      <td>1.573237</td>
      <td>64.798208</td>
      <td>2283.300441</td>
      <td>0.265785</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.500014</td>
      <td>0.368844</td>
      <td>0.499729</td>
      <td>0.457629</td>
      <td>24.545260</td>
      <td>0.295571</td>
      <td>0.948627</td>
      <td>0.737271</td>
      <td>0.859962</td>
      <td>0.880394</td>
      <td>0.880178</td>
      <td>0.861674</td>
      <td>0.885285</td>
      <td>0.885385</td>
      <td>0.832934</td>
      <td>0.491363</td>
      <td>1.067504</td>
      <td>30.085974</td>
      <td>2266.771362</td>
      <td>0.441782</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>18.250000</td>
      <td>18.800000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>35.587500</td>
      <td>401.450000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>29.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>70.350000</td>
      <td>1397.475000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>55.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>89.862500</td>
      <td>3794.737500</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>72.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>118.750000</td>
      <td>8684.800000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Looks like all of the columns are now numerical and ready for further exploration and modelling.

### Visualization
Let us further analyze data through visualizations.


```python
### Function for KDE plots
def kdeplot_churn(col):
    ## Set size of figure 
    plt.figure(figsize=(20,7))

    ## KDE plots for each category label using Seaborn
    sns.kdeplot(data=df_clean[df_clean["Churn"]==0][col], 
                label="Churn - No", shade=True)
    sns.kdeplot(data=df_clean[df_clean["Churn"]==1][col], 
                label="Churn - Yes", shade=True)

    ## Label x ticks
    if not col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        plt.xticks( np.arange(len(le[col].classes_)), (le[col].classes_) )
    
    ## Add title
    plt.title("DISTRIBUTION OF {} BY CHURN".format(col.upper()))
```

Shown below are the KDE plots and category plots of each predictor column versus the target variable `Churn`. From these plots, we can see how balanced/imbalanced the datas are, and which categories likely classify `Churn`.


```python
### KDE plot to see distributions by churn
for col in df_clean.columns:
    if col != "Churn":
        kdeplot_churn(col)
        if not col in ["MonthlyCharges", "TotalCharges", "tenure"]:
            plt.figure(figsize=(20,7))
            sns.catplot(x=col, kind="count", hue="Churn", palette="ch:.25", data=df_clean)
```


![png](Images\output_29_0.png)



    <Figure size 1440x504 with 0 Axes>



![png](Images\output_29_2.png)



![png](Images\output_29_3.png)



    <Figure size 1440x504 with 0 Axes>



![png](Images\output_29_5.png)



![png](Images\output_29_6.png)



    <Figure size 1440x504 with 0 Axes>



![png](Images\output_29_8.png)



![png](Images\output_29_9.png)



    <Figure size 1440x504 with 0 Axes>



![png](Images\output_29_11.png)



![png](Images\output_29_12.png)



![png](Images\output_29_13.png)



    <Figure size 1440x504 with 0 Axes>



![png](Images\output_29_15.png)



![png](Images\output_29_16.png)



    <Figure size 1440x504 with 0 Axes>



![png](Images\output_29_18.png)



![png](Images\output_29_19.png)



    <Figure size 1440x504 with 0 Axes>



![png](Images\output_29_21.png)



![png](Images\output_29_22.png)



    <Figure size 1440x504 with 0 Axes>



![png](Images\output_29_24.png)



![png](Images\output_29_25.png)



    <Figure size 1440x504 with 0 Axes>



![png](Images\output_29_27.png)



![png](Images\output_29_28.png)



    <Figure size 1440x504 with 0 Axes>



![png](Images\output_29_30.png)



![png](Images\output_29_31.png)



    <Figure size 1440x504 with 0 Axes>



![png](Images\output_29_33.png)



![png](Images\output_29_34.png)



    <Figure size 1440x504 with 0 Axes>



![png](Images\output_29_36.png)



![png](Images\output_29_37.png)



    <Figure size 1440x504 with 0 Axes>



![png](Images\output_29_39.png)



![png](Images\output_29_40.png)



    <Figure size 1440x504 with 0 Axes>



![png](Images\output_29_42.png)



![png](Images\output_29_43.png)



    <Figure size 1440x504 with 0 Axes>



![png](Images\output_29_45.png)



![png](Images\output_29_46.png)



    <Figure size 1440x504 with 0 Axes>



![png](Images\output_29_48.png)



![png](Images\output_29_49.png)



![png](Images\output_29_50.png)


## Prepare the Data for Modelling
Now, let us prepare the data for modelling.

### Train-Test Split
The dataset is divided into train set and test set.

The train set is used for building the model, while the test set is used for validating the model.


```python
### Separate the predictors from the target variable
X = df_clean.drop(["Churn"], axis=1)
y = df_clean["Churn"]

print("Size of x (predictors):\t{}\nSize of y (target):\t{}".format(X.shape, y.shape))
```

    Size of x (predictors):	(7032, 19)
    Size of y (target):	(7032,)
    


```python
### Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)

### Check shape to make sure it is all in order
print("Size of x_train: {} \t Size of x_test: {} \nSize of y_train: {} \t Size of y_test: {}".format(
    X_train.shape, X_test.shape, y_train.shape, y_test.shape))
```

    Size of x_train: (4922, 19) 	 Size of x_test: (2110, 19) 
    Size of y_train: (4922,) 	 Size of y_test: (2110,)
    


```python
print(y_train.value_counts(), '\n', y_test.value_counts())
```

    0    3614
    1    1308
    Name: Churn, dtype: int64 
     0    1549
    1     561
    Name: Churn, dtype: int64
    

Shown above is the distribution of `Churn` data, and we can see that it is imbalanced. The dataset is dominated by rows where `Churn == 0`, and we can make it more balanced using the **resampling technique** performed below.

### Resample


```python
### Concatenate the train data before resampling 
df_train = pd.concat([X_train, y_train], axis=1)
print("DF Train shape:", df_train.shape, "\nDF Train value counts:\n",df_train['Churn'].value_counts())
df_train.head()
```

    DF Train shape: (4922, 20) 
    DF Train value counts:
     0    3614
    1    1308
    Name: Churn, dtype: int64
    





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>OnlineBackup</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
    <tr>
      <th>customerID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7105-MXJLL</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>26</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>60.70</td>
      <td>1597.40</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6754-WKSHP</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>30</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>25.35</td>
      <td>723.30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1984-FCOWB</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>70</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>109.50</td>
      <td>7674.55</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5188-HGMLP</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>54</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>74.00</td>
      <td>3919.15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0196-JTUQI</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>75.20</td>
      <td>633.85</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
### Append the category with highest value counts
df_train_res = df_train[df_train["Churn"]==0]

### Resample minority class
resampled = resample(df_train[df_train["Churn"]==1],
                    replace=True, # sample with replacement
                    n_samples=3614, # match number in majority class
                    random_state=1) # reproducible results
df_train_res = pd.concat([df_train_res, resampled]) 

### Print resampled training set to check
print("Size of df_train_res:", df_train_res.shape, "\nValue counts for Churn:\n", 
      df_train_res["Churn"].value_counts())
df_train_res.head()
```

    Size of df_train_res: (7228, 20) 
    Value counts for Churn:
     1    3614
    0    3614
    Name: Churn, dtype: int64
    





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>OnlineBackup</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
    <tr>
      <th>customerID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7105-MXJLL</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>26</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>60.70</td>
      <td>1597.40</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6754-WKSHP</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>30</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>25.35</td>
      <td>723.30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5188-HGMLP</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>54</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>74.00</td>
      <td>3919.15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0196-JTUQI</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>75.20</td>
      <td>633.85</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8755-IWJHN</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>69</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>95.35</td>
      <td>6382.00</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train_res = df_train_res.drop(["Churn"], axis=1)
y_train_res = df_train_res["Churn"]

print("Size of x_train_res: {}\nSize of y_train_res: {}".format(X_train_res.shape, y_train_res.shape))
```

    Size of x_train_res: (7228, 19)
    Size of y_train_res: (7228,)
    

Eventhough we have resampled data, we will still consider the original data and see which of those datasets enhances model performance.

## Build the Models

Now that our data is prepared for modelling, let us initialize functions that we will use for it.


```python
### Function for computing for the recall of Churn==0
def recall_0(correct, pred):
    return metrics.recall_score(y_true=correct, y_pred=pred, pos_label=0, average="binary")

### Function for getting and displaying score
def score_card(model, predicted):
    ## Print classification report
    print( "Classification report for {}:\n{}".format( model, 
                                                metrics.classification_report(y_test, predicted) ) )

    ## Cross-Validation
    scoring = {
        "recall_0": metrics.make_scorer(recall_0),
        "accuracy": "accuracy",
        "recall": "recall",
        "precision": "precision",
        "f1": "f1"
    }
    cv_scores = cross_validate(model, X, y, scoring=scoring, cv=10)
    
    ## Return scores
    return { c: np.mean(cv_scores[c]) for c in cv_scores }
```


```python
### Initialized for easy plotting of confusion matrix
def confmatrix(y_pred, title):
    cm = metrics.confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, columns=np.unique(y_test), index = np.unique(y_test))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    
    plt.figure(figsize = (10,7))
    plt.title(title)
    
    sns.set(font_scale=1.4) # For label size
    sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16}) # Font size
```

### Pipeline
For quickly choosing the best model, we can construct pipelines for each of the machine algorithms:
* Logistic Regression
* K-Nearest Neighbors
* Naive Bayes
* Decision Trees
* Random Forest Classifier
* XGBoost

We'll also use two types datasets per algorithm:
* Resampled Data
* Original Data

The top 3 models with the highest **accuracy scores** are used for further modelling and tuning.


```python
### Make pipelines
pipelines = {
    "LogReg": make_pipeline(StandardScaler(), LogisticRegression()),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "GNB": make_pipeline(StandardScaler(), GaussianNB()),
    "DTree": make_pipeline(StandardScaler(), DecisionTreeClassifier()),
    "RF": make_pipeline(StandardScaler(), RandomForestClassifier()),
    "XGB": make_pipeline(StandardScaler(), GradientBoostingClassifier())
}
```


```python
### Initialize dict for results
pipe_results = {}

### Iterate through pipelines to get each accuracy score
for pipe in pipelines:
    model_orig = pipelines[pipe]
    model_orig.fit(X_train, y_train)
    
    pipe_results[pipe] = [metrics.accuracy_score(y_test, model_orig.predict(X_test)) * 100]
    
    model_res = pipelines[pipe]
    model_res.fit(X_train_res, y_train_res)
    
    pipe_results[pipe].append(metrics.accuracy_score(y_test, model_res.predict(X_test)) * 100)

### Print accuracy scores got per pipeline
print("-------Accuracy Scores per Pipeline-------")
print("Algorithm\tOriginal\tResampled")
for pipe in pipe_results:
    print( "%s\t\t%.3f\t\t%.3f" % (pipe, pipe_results[pipe][0], pipe_results[pipe][1]) )
```

    -------Accuracy Scores per Pipeline-------
    Algorithm	Original	Resampled
    LogReg		80.142		73.602
    KNN		75.498		67.156
    GNB		74.645		72.986
    DTree		71.801		72.085
    RF		78.199		77.346
    XGB		80.332		74.882
    

From the results above, the top 3 models are:
1. XGBoost Classifier
1. Logistic Regression
1. Random Forest Classifier

Below, we'll try to tune these models using `GridSearchCV` and find out which ones are the best for this use case. Note that not all of the parameters will be tuned due to limits in memory.

### Random Forest Classifier

#### Build/Train the Model
##### Find Best Parameters


```python
rf_parameters = {
    "randomforestclassifier__n_estimators" : [n for n in np.arange(100,800,200)], 
    "randomforestclassifier__criterion" : ["gini", "entropy"]
}

rf_grid = GridSearchCV(pipelines["RF"], param_grid = rf_parameters,
                           cv = 5, scoring="accuracy")

rf_grid.fit(X, y)
print("-----RF GridSearch-----")
print( "Best Params: {}\nBest Score: {}".format(rf_grid.best_params_, rf_grid.best_score_) )
```

    -----RF GridSearch-----
    Best Params: {'randomforestclassifier__criterion': 'entropy', 'randomforestclassifier__n_estimators': 100}
    Best Score: 0.7945108077360638
    

##### Use best parameters


```python
### Instantiate the algorithm
rf = RandomForestClassifier(criterion="entropy", n_estimators=100)

### Fit the model
rf.fit(X_train, y_train)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)



#### Validate the Model

##### Classification Report


```python
### Predict on test set
rf_pred = rf.predict(X_test)

### Get score report for model
rf_score = score_card(rf, rf_pred)
pd.DataFrame.from_dict({"Cross-Validation Scores":rf_score})
```

    Classification report for RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False):
                  precision    recall  f1-score   support
    
               0       0.83      0.90      0.86      1549
               1       0.64      0.49      0.55       561
    
        accuracy                           0.79      2110
       macro avg       0.73      0.69      0.71      2110
    weighted avg       0.78      0.79      0.78      2110
    
    





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cross-Validation Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fit_time</th>
      <td>8.169977</td>
    </tr>
    <tr>
      <th>score_time</th>
      <td>0.652181</td>
    </tr>
    <tr>
      <th>test_accuracy</th>
      <td>0.794935</td>
    </tr>
    <tr>
      <th>test_f1</th>
      <td>0.563180</td>
    </tr>
    <tr>
      <th>test_precision</th>
      <td>0.650402</td>
    </tr>
    <tr>
      <th>test_recall</th>
      <td>0.497059</td>
    </tr>
    <tr>
      <th>test_recall_0</th>
      <td>0.902765</td>
    </tr>
  </tbody>
</table>
</div>



##### Confusion Matrix


```python
### Plot the confusion matrix
confmatrix(rf_pred, "Random Forest - Telco Customer Churn\nConfusion Matrix")
```


![png](Images\output_58_0.png)


### Logistic Regression

#### Build/Train the Model
##### Find Best Parameters


```python
logreg_parameters = {
    "logisticregression__penalty" : ["l1", "l2"], 
    "logisticregression__C" : [n for n in np.arange(0.5,3,0.5)]
}

logreg_grid = GridSearchCV(pipelines["LogReg"], param_grid = logreg_parameters,
                           cv = 5, scoring="accuracy")

logreg_grid.fit(X, y)
print("-----LogReg GridSearch-----")
print( "Best Params: {}\nBest Score: {}".format(logreg_grid.best_params_, logreg_grid.best_score_) )
```

    -----LogReg GridSearch-----
    Best Params: {'logisticregression__C': 0.5, 'logisticregression__penalty': 'l1'}
    Best Score: 0.801905574516496
    

##### Use best parameters


```python
### Instantiate the algorithm
logreg = LogisticRegression(C=0.5, penalty="l1")

### Fit the model
logreg.fit(X_train, y_train)
```




    LogisticRegression(C=0.5, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l1',
                       random_state=None, solver='warn', tol=0.0001, verbose=0,
                       warm_start=False)



#### Validate the Model

##### Classification Report


```python
### Predict on test set
logreg_pred = logreg.predict(X_test)

### Get score report for model
logreg_score = score_card(logreg, logreg_pred)
pd.DataFrame.from_dict({"Cross-Validation Scores":logreg_score})
```

    Classification report for LogisticRegression(C=0.5, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l1',
                       random_state=None, solver='warn', tol=0.0001, verbose=0,
                       warm_start=False):
                  precision    recall  f1-score   support
    
               0       0.85      0.89      0.87      1549
               1       0.65      0.56      0.60       561
    
        accuracy                           0.80      2110
       macro avg       0.75      0.73      0.73      2110
    weighted avg       0.79      0.80      0.80      2110
    
    





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cross-Validation Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fit_time</th>
      <td>0.696976</td>
    </tr>
    <tr>
      <th>score_time</th>
      <td>0.076322</td>
    </tr>
    <tr>
      <th>test_accuracy</th>
      <td>0.801480</td>
    </tr>
    <tr>
      <th>test_f1</th>
      <td>0.592991</td>
    </tr>
    <tr>
      <th>test_precision</th>
      <td>0.650664</td>
    </tr>
    <tr>
      <th>test_recall</th>
      <td>0.545219</td>
    </tr>
    <tr>
      <th>test_recall_0</th>
      <td>0.894247</td>
    </tr>
  </tbody>
</table>
</div>



##### Confusion Matrix


```python
### Plot the confusion matrix
confmatrix(logreg_pred, "LogReg - Telco Customer Churn\nConfusion Matrix")
```


![png](Images\output_68_0.png)


### XGBoost

#### Build/Train the Model
##### Find Best Parameters


```python
xgb_parameters = {
    "gradientboostingclassifier__loss" : ["deviance", "exponential"], 
    "gradientboostingclassifier__n_estimators" : [n for n in np.arange(100,800,200)]
}

xgb_grid = GridSearchCV(pipelines["XGB"], param_grid = xgb_parameters,
                           cv = 5, scoring="accuracy")

xgb_grid.fit(X, y)
print("-----XGB GridSearch-----")
print( "Best Params: {}\nBest Score: {}".format(xgb_grid.best_params_,xgb_grid.best_score_) )
```

    -----XGB GridSearch-----
    Best Params: {'gradientboostingclassifier__loss': 'deviance', 'gradientboostingclassifier__n_estimators': 100}
    Best Score: 0.8056029579067122
    

##### Use best parameters


```python
### Instantiate the algorithm
xgb = GradientBoostingClassifier(loss="deviance", n_estimators=100)

### Fit the model
xgb.fit(X_train, y_train)
```




    GradientBoostingClassifier(criterion='friedman_mse', init=None,
                               learning_rate=0.1, loss='deviance', max_depth=3,
                               max_features=None, max_leaf_nodes=None,
                               min_impurity_decrease=0.0, min_impurity_split=None,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=100,
                               n_iter_no_change=None, presort='auto',
                               random_state=None, subsample=1.0, tol=0.0001,
                               validation_fraction=0.1, verbose=0,
                               warm_start=False)



#### Validate the Model

##### Classification Report


```python
### Predict on test set
xgb_pred = xgb.predict(X_test)

### Get score report for model
xgb_score = score_card(xgb, xgb_pred)
pd.DataFrame.from_dict({"Cross-Validation Scores":xgb_score})
```

    Classification report for GradientBoostingClassifier(criterion='friedman_mse', init=None,
                               learning_rate=0.1, loss='deviance', max_depth=3,
                               max_features=None, max_leaf_nodes=None,
                               min_impurity_decrease=0.0, min_impurity_split=None,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=100,
                               n_iter_no_change=None, presort='auto',
                               random_state=None, subsample=1.0, tol=0.0001,
                               validation_fraction=0.1, verbose=0,
                               warm_start=False):
                  precision    recall  f1-score   support
    
               0       0.84      0.91      0.87      1549
               1       0.67      0.51      0.58       561
    
        accuracy                           0.80      2110
       macro avg       0.75      0.71      0.73      2110
    weighted avg       0.79      0.80      0.79      2110
    
    





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cross-Validation Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fit_time</th>
      <td>6.034562</td>
    </tr>
    <tr>
      <th>score_time</th>
      <td>0.133802</td>
    </tr>
    <tr>
      <th>test_accuracy</th>
      <td>0.803471</td>
    </tr>
    <tr>
      <th>test_f1</th>
      <td>0.583099</td>
    </tr>
    <tr>
      <th>test_precision</th>
      <td>0.669511</td>
    </tr>
    <tr>
      <th>test_recall</th>
      <td>0.517388</td>
    </tr>
    <tr>
      <th>test_recall_0</th>
      <td>0.907029</td>
    </tr>
  </tbody>
</table>
</div>



##### Confusion Matrix


```python
### Plot the confusion matrix
confmatrix(xgb_pred, "XGBoost - Telco Customer Churn\nConfusion Matrix")
```


![png](Images\output_78_0.png)


## Summary of the Results


```python
### Compile results into dataframe
df_results = pd.DataFrame.from_dict({
    "Logistic Regression": logreg_score, "XGBoost": xgb_score,
    "Random Forest": rf_score}, orient="index")

### Convert scores into percentages
for m in ["test_recall_0", "test_accuracy", "test_recall", "test_precision", "test_f1"]:
    df_results[m] = df_results[m] * 100

### Show resulting dataframe
df_results
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fit_time</th>
      <th>score_time</th>
      <th>test_recall_0</th>
      <th>test_accuracy</th>
      <th>test_recall</th>
      <th>test_precision</th>
      <th>test_f1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Logistic Regression</th>
      <td>0.696976</td>
      <td>0.076322</td>
      <td>89.424715</td>
      <td>80.148018</td>
      <td>54.521879</td>
      <td>65.066414</td>
      <td>59.299086</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>8.169977</td>
      <td>0.652181</td>
      <td>90.276528</td>
      <td>79.493538</td>
      <td>49.705882</td>
      <td>65.040211</td>
      <td>56.317993</td>
    </tr>
    <tr>
      <th>XGBoost</th>
      <td>6.034562</td>
      <td>0.133802</td>
      <td>90.702922</td>
      <td>80.347064</td>
      <td>51.738830</td>
      <td>66.951062</td>
      <td>58.309915</td>
    </tr>
  </tbody>
</table>
</div>




```python
### Find out the best model based on all metrics
print("-----Best Model per Metric-----")

## Initiate dict of best model pre metric
best = {"test_recall_0":0, "test_accuracy":0, "test_recall":0, "test_precision":0, "test_f1":0}

## Iterate through the dict and columns of df_results to find the max value and index
for m in best:
    best[m] = { "Model":df_results[m].idxmax(), "Score":df_results[m].max() }

## Display the results
df_best = pd.DataFrame.from_dict(best, orient="index")
print(df_best["Model"].value_counts()[:1])
df_best
```

    -----Best Model per Metric-----
    XGBoost    3
    Name: Model, dtype: int64
    





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>test_accuracy</th>
      <td>XGBoost</td>
      <td>80.347064</td>
    </tr>
    <tr>
      <th>test_f1</th>
      <td>Logistic Regression</td>
      <td>59.299086</td>
    </tr>
    <tr>
      <th>test_precision</th>
      <td>XGBoost</td>
      <td>66.951062</td>
    </tr>
    <tr>
      <th>test_recall</th>
      <td>Logistic Regression</td>
      <td>54.521879</td>
    </tr>
    <tr>
      <th>test_recall_0</th>
      <td>XGBoost</td>
      <td>90.702922</td>
    </tr>
  </tbody>
</table>
</div>



In this exercise, we tried categorical visualizations from `Seaborn` which are: 
* `catplot` - like a bar plot but for categories
* `kdeplot` - like a smoothened histogram

We also explored `SKLearn`'s `make_pipeline` which is very useful in saving time and typing when modelling.

Aside from that, we tried `GridSearchCV` to find the best parameters to use per model based on a dictionary of lists we defined.

For this data, we would want high recall to better detecting customers who are in risk of churning. **Logistic Regression may have given the best score for recall but that score is still low.**

So in this case, **XGBoost is still the best model** because eventhough it has less recall, **XGBoost gave the best performance for most of the metrics**. It has **high accuracy, precision and recall(for 0).** 

## Special Thanks
* [FTW Foundation](https://ftwfoundation.org)
