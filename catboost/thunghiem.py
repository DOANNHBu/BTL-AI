import pandas as pd
import pandas as pd
import os
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
file_path = 'D:\\1 code AI'
file_names = ['new_data_v1.csv']
datasets = [pd.read_csv(os.path.join(file_path, file)) for file in file_names]
dataset = pd.concat(datasets, ignore_index=True)

# Count the number of labels 0 and 1
label_counts = dataset['Label'].value_counts()
print("Label counts:")
print(label_counts)

# Display dataset info
dataset.info()

# Check if columns exist before dropping
columns_drop = ["Label", "year", "month", "day", "hour"]
existing_columns_to_drop = [col for col in columns_drop if col in dataset.columns]

X = dataset.drop(columns=existing_columns_to_drop)
y = dataset["Label"]

# Display feature info
print("Feature data (X) preview:")
print(X.head())
print("Feature data (X) info:")
print(X.info())
print("Labels (y) preview:")
print(y.head())
print("Label distribution:")
print(y.value_counts())

# Split the data into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply smote
smote = SMOTE(sampling_strategy=0.2,random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

model = CatBoostClassifier(verbose=3)
model.fit(X_train_resampled, y_train_resampled)
y_pred = model.predict(X_test)

print("Classification Report - Class Weights:")
print(classification_report(y_test, y_pred))
