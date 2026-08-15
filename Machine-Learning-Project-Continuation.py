import pandas as pd
from sklearn.compose import  ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN


data = pd.read_csv("student_performance_dataset.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# print(data.columns)
# print(data.describe())
data = data.drop("student_id", axis=1)
# print(data.isnull().sum())
data.drop_duplicates(inplace=True)
data.reset_index(drop=True, inplace=True)
# print(data.head())
# data.info()

# print(data['final_grade'].unique())

Contain_number = [
    'study_time_hours',
    'attendance_percent',
    'sleep_hours',
    'previous_grade',
    'final_exam_score'
]

Contain_character = [
    'gender',
    'parental_education',
    'internet_access',
    'extracurricular_activities',
    'part_time_job',

]
# Converting Columns containing characters into a computer understandable format
preprocessing  = ColumnTransformer(
    transformers=[
        (
            'Contain_character',
             OneHotEncoder(handle_unknown='ignore'),
             Contain_character
        ),
        (
            'Contain_number',
            StandardScaler(),
            Contain_number
        )

    ])

# Training the model using train test spit

x = data.drop("final_grade", axis=1)
y = data['final_grade'].map({
    'A' :0,
    'B' :1,
    'C' :2,
    'D' :3,
    'F' :4
})

# Splitting the model using train test split
x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Training with Decision Tree
decision_tree = Pipeline(
    steps=[
        ('preprocessing', preprocessing),
        ('decision_tree',DecisionTreeClassifier(max_depth=10,class_weight='balanced'))
    ])

# Training with Random Forest
random_forest = Pipeline(
    steps=[
        ('preprocessing', preprocessing),
        ('random_forest',RandomForestClassifier(n_estimators=100, max_depth=10,class_weight='balanced')),
    ])

# Training with Logistic Classifier
logistic_classifier = Pipeline(
    steps=[
        ('preprocessing', preprocessing),
        ('logistic_classifier',LogisticRegression(class_weight='balanced'))
    ]
)

# Training with XGB
XGboost_classifier = Pipeline(
    steps=[
        ('preprocessing', preprocessing),
        ("XGboost",XGBClassifier(n_estimators=100, max_depth=10))
    ]
)

# Fitting, Predicting and Calculating  the Accuracy of the  models

decision_tree.fit(x_train,y_train)
random_forest.fit(x_train,y_train)
logistic_classifier.fit(x_train,y_train)
XGboost_classifier.fit(x_train,y_train)

decision = decision_tree.predict(x_test)
random = random_forest.predict(x_test)
logistic = logistic_classifier.predict(x_test)
XGboost = XGboost_classifier.predict(x_test)

decision_accuracy = accuracy_score(y_test, decision)
random_accuracy = accuracy_score(y_test, random)
logistic_accuracy = accuracy_score(y_test, logistic)
XGboost_accuracy = accuracy_score(y_test, XGboost)

print("\n")
print(f"Random Accuracy: {random_accuracy}")
print(f"Random forest: {random}")
print("\n")
print(f"Decision tree: {decision}")
print(f"Decision Accuracy: {decision_accuracy}")
print("\n")
print(f"Logistic classifier: {logistic}")
print(f"Logistic Accuracy: {logistic_accuracy}")
print("\n")
print(f"XGboost classifier: {XGboost}")
print(f"XGboost Accuracy: {XGboost_accuracy}")
print("\n")

# Trying an unsupervised learning

kmeans = Pipeline(steps=[
    ('preprocessing', preprocessing),
    ("Kmeans",KMeans(n_clusters=10)),
])

kmeans.fit(x)

data['Kmeans columns'] = kmeans.named_steps['Kmeans'].labels_

agglomerative = Pipeline(steps=[
    ('preprocessing', preprocessing),
    ("Agglomerative",AgglomerativeClustering(n_clusters=10)),
])
agglomerative.fit(x)

data['Hierarchical column'] = agglomerative.named_steps['Agglomerative'].labels_

dbscan = Pipeline(steps=[
    ('preprocessing', preprocessing),
    ("DBSCAN",DBSCAN()),
])
dbscan.fit(x)

data['DBSCAN column'] = dbscan.named_steps['DBSCAN'].labels_
print(data.head(20))

import os
print(os.getcwd())