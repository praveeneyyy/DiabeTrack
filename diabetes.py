#!/usr/bin/env python
# coding: utf-8

# # Description:                                                                                                                   
# The objective of the dataset is to predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.
# 
# Dataset url : https://www.kaggle.com/uciml/pima-indians-diabetes-database

# # Step 0: Import libraries and Dataset

# In[1]:


# Importing libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Importing dataset
dataset = pd.read_csv('diabetes.csv')


# # Step 1: Descriptive Statistics

# In[3]:


# Preview data
dataset.head()


# In[4]:


# Dataset dimensions - (rows, columns)
dataset.shape


# In[5]:


# Features data-type
dataset.info()


# In[6]:


# Statistical summary
dataset.describe().T


# In[7]:


# Count of null values
dataset.isnull().sum()


# ## Observations:
# 1. There are a total of 768 records and 9 features in the dataset.
# 2. Each feature can be either of integer or float dataype.
# 3. Some features like Glucose, Blood pressure , Insulin, BMI have zero values which represent missing data.
# 4. There are zero NaN values in the dataset.
# 5. In the outcome column, 1 represents diabetes positive and 0 represents diabetes negative.

# # Step 2: Data Visualization

# In[8]:


# Outcome countplot
sns.countplot(x = 'Outcome',data = dataset)


# In[9]:


# Histogram of each feature
import itertools

col = dataset.columns[:8]
plt.subplots(figsize = (20, 15))
length = len(col)

for i, j in itertools.zip_longest(col, range(length)):
    plt.subplot((length/2), 3, j + 1)
    plt.subplots_adjust(wspace = 0.1,hspace = 0.5)
    dataset[i].hist(bins = 20)
    plt.title(i)
plt.show()


# In[10]:


# Scatter plot matrix 
from pandas.tools.plotting import scatter_matrix
scatter_matrix(dataset, figsize = (20, 20));


# In[11]:


# Pairplot 
sns.pairplot(data = dataset, hue = 'Outcome')
plt.show()


# In[12]:


# Heatmap
sns.heatmap(dataset.corr(), annot = True)
plt.show()


# ## Observations:
# 1. The countplot tells us that the dataset is imbalanced, as number of patients who don't have diabetes is more than those who do.
# 2. From the correaltion heatmap, we can see that there is a high correlation between Outcome and [Glucose,BMI,Age,Insulin]. We can select these features to accept input from the user and predict the outcome.

# # Step 3: Data Preprocessing

# In[13]:


dataset_new = dataset


# In[14]:


# Replacing zero values with NaN
dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN) 


# In[15]:


# Count of NaN
dataset_new.isnull().sum()


# In[16]:


# Replacing NaN with mean values
dataset_new["Glucose"].fillna(dataset_new["Glucose"].mean(), inplace = True)
dataset_new["BloodPressure"].fillna(dataset_new["BloodPressure"].mean(), inplace = True)
dataset_new["SkinThickness"].fillna(dataset_new["SkinThickness"].mean(), inplace = True)
dataset_new["Insulin"].fillna(dataset_new["Insulin"].mean(), inplace = True)
dataset_new["BMI"].fillna(dataset_new["BMI"].mean(), inplace = True)


# In[17]:


# Statistical summary
dataset_new.describe().T


# In[18]:


# Feature scaling using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
dataset_scaled = sc.fit_transform(dataset_new)


# In[19]:


dataset_scaled = pd.DataFrame(dataset_scaled)


# In[20]:


# Selecting features - [Glucose, Insulin, BMI, Age]
X = dataset_scaled.iloc[:, [1, 4, 5, 7]].values
Y = dataset_scaled.iloc[:, 8].values


# In[21]:


# Splitting X and Y
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = dataset_new['Outcome'] )


# In[22]:


# Checking dimensions
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)


# # Step 4: Data Modelling

# In[23]:


# Logistic Regression Algorithm
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state = 42)
logreg.fit(X_train, Y_train)


# In[24]:


# Plotting a graph for n_neighbors 
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

X_axis = list(range(1, 31))
acc = pd.Series()
x = range(1,31)

for i in list(range(1, 31)):
    knn_model = KNeighborsClassifier(n_neighbors = i) 
    knn_model.fit(X_train, Y_train)
    prediction = knn_model.predict(X_test)
    acc = acc.append(pd.Series(metrics.accuracy_score(prediction, Y_test)))
plt.plot(X_axis, acc)
plt.xticks(x)
plt.title("Finding best value for n_estimators")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.grid()
plt.show()
print('Highest value: ',acc.values.max())


# In[25]:


# K nearest neighbors Algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 24, metric = 'minkowski', p = 2)
knn.fit(X_train, Y_train)


# In[26]:


# Support Vector Classifier Algorithm
from sklearn.svm import SVC
svc = SVC(kernel = 'linear', random_state = 42)
svc.fit(X_train, Y_train)


# In[27]:


# Naive Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, Y_train)


# In[28]:


# Decision tree Algorithm
from sklearn.tree import DecisionTreeClassifier
dectree = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
dectree.fit(X_train, Y_train)


# In[29]:


# Random forest Algorithm
from sklearn.ensemble import RandomForestClassifier
ranfor = RandomForestClassifier(n_estimators = 11, criterion = 'entropy', random_state = 42)
ranfor.fit(X_train, Y_train)


# In[30]:


# Making predictions on test dataset
Y_pred_logreg = logreg.predict(X_test)
Y_pred_knn = knn.predict(X_test)
Y_pred_svc = svc.predict(X_test)
Y_pred_nb = nb.predict(X_test)
Y_pred_dectree = dectree.predict(X_test)
Y_pred_ranfor = ranfor.predict(X_test)


# # Step 5: Model Evaluation

# In[31]:


# Evaluating using accuracy_score metric
from sklearn.metrics import accuracy_score
accuracy_logreg = accuracy_score(Y_test, Y_pred_logreg)
accuracy_knn = accuracy_score(Y_test, Y_pred_knn)
accuracy_svc = accuracy_score(Y_test, Y_pred_svc)
accuracy_nb = accuracy_score(Y_test, Y_pred_nb)
accuracy_dectree = accuracy_score(Y_test, Y_pred_dectree)
accuracy_ranfor = accuracy_score(Y_test, Y_pred_ranfor)


# In[32]:


# Accuracy on test set
print("Logistic Regression: " + str(accuracy_logreg * 100))
print("K Nearest neighbors: " + str(accuracy_knn * 100))
print("Support Vector Classifier: " + str(accuracy_svc * 100))
print("Naive Bayes: " + str(accuracy_nb * 100))
print("Decision tree: " + str(accuracy_dectree * 100))
print("Random Forest: " + str(accuracy_ranfor * 100))


# In[33]:


#From the above comparison, we can observe that K Nearest neighbors gets the highest accuracy of 78.57 %


# In[34]:


# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred_knn)
cm


# In[35]:


# Heatmap of Confusion matrix
sns.heatmap(pd.DataFrame(cm), annot=True)


# In[36]:


# Classification report
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred_knn))

