from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

# Load your dataset
file_path = 'D:\\1 code AI'
file_names = ['new_data_v1.csv']
datasets = [pd.read_csv(os.path.join(file_path, file)) for file in file_names]
dataset = pd.concat(datasets, ignore_index=True)

# Count the number of labels 0 and 1
label_counts = dataset['Label'].value_counts()
print("Label counts:")
print(label_counts)

# Prepare the data
columns_drop = ["Label", "Label", "year", "month", "day", "hour"]
existing_columns_to_drop = [col for col in columns_drop if col in dataset.columns]
X = dataset.drop(columns=existing_columns_to_drop)
y = dataset["Label"]

# Split the data into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance the training data
smote = SMOTE(sampling_strategy=0.2, random_state=42)  # 20% of majority class
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Check the new label distribution after SMOTE
print("Label distribution after SMOTE:")
print(pd.Series(y_train_smote).value_counts())

print(f"Training set size: {len(X_train_smote)}")
print(f"Test set size: {len(X_test)}")

print("Training set label distribution:")
print(pd.Series(y_train_smote).value_counts())
print("Test set label distribution:")
print(pd.Series(y_test).value_counts())

# Initialize and train the RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=900,
    min_samples_split=11,
    min_samples_leaf=4,
    max_features=None,
    max_depth=50,
    class_weight='balanced',
    random_state=42,
    n_jobs=10
)

rf.fit(X_train_smote, y_train_smote)

# Predict and print classification report
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))