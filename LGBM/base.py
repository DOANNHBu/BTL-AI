import pandas as pd
import os
from imblearn.over_sampling import SMOTE

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
columns_drop = ["Label", "row", "col", "year", "month", "day", "hour"]
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

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print("Training set label distribution:")
print(y_train.value_counts())
print("Test set label distribution:")
print(y_test.value_counts())

# params = {
#     'learning_rate': 0.15,
#     #'is_unbalance': 'true', # replaced with scale_pos_weight argument
#     'num_leaves': 7,  # 2^max_depth - 1
#     'max_depth': 3,  # -1 means no limit
#     'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
#     'max_bin': 100,  # Number of bucketed bin for feature values
#     'subsample': 0.7,  # Subsample ratio of the training instance.
#     'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
#     'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
#     'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
#     'scale_pos_weight':99 # because training data is extremely unbalanced 
# }

# # Initialize and train LightGBM Classifier
# clf = lgb.LGBMClassifier(boosting_type= 'gbdt'  , n_estimators=1000,
#                           objective='binary', n_jobs=-1, random_state=42, metrics = 'f1',num_boost_round = 500)
# clf.fit(X_train, y_train)


# Apply SMOTE to the training data
smote = SMOTE(sampling_strategy=0.3, k_neighbors=20, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(y_train.value_counts())
print(y_train_res.value_counts())

params = {
    'learning_rate': 0.2,
    'num_leaves': 200,
    'min_child_samples': 250,
    'max_bin': 200,
    'subsample': 1,
    'subsample_freq': 1,
    'colsample_bytree': 0.9,
    'min_child_weight': 0,
    'scale_pos_weight': 5.71, # label 1 is 17.5% of the data so the scale_pos_weight is 1/0.175 = 5.71
    'device_type': 'gpu'
}

# Initialize and train LightGBM Classifier with unpacked parameters
clf = lgb.LGBMClassifier(boosting_type='gbdt', n_estimators=2000, objective='binary', 
                         n_jobs=-1, random_state=42, **params)
clf.fit(X_train_res, y_train_res)

# Make predictions
pred_labels = clf.predict(X_test)


clf.fit(X_train, y_train)


# Print classification report
print("Classification Report:")
print(classification_report(y_test, pred_labels))

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, pred_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
