import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report

# Load the dataset
file_path = 'D:\\1 code AI'
file_names = ['new_data_v1.csv']
datasets = [pd.read_csv(os.path.join(file_path, file)) for file in file_names]
dataset = pd.concat(datasets, ignore_index=True)

# Count the number of labels 0 and 1
label_counts = dataset['Label'].value_counts()
print("Label counts:")
print(label_counts)
dataset.info()

# Define features (X) and target (y)
columns_drop = ["Label", "year", "month", "day", "hour"]
existing_columns_to_drop = [col for col in columns_drop if col in dataset.columns]

X = dataset.drop(columns=existing_columns_to_drop)
y = dataset["Label"]

# Split the data into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define the base LightGBM models
base_estimators = [
    (f'lightgbm_{i}', LGBMClassifier(n_estimators=1000, random_state=42 + i))
    for i in range(8)
]

# Create the Stacking Classifier with LightGBM base models
stacking_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LGBMClassifier(n_estimators=1000, random_state=42),
    cv=5  # Cross-validation splits for stacking
)

# Train the Stacking Classifier
stacking_clf.fit(X_train, y_train)

# Make predictions
y_pred = stacking_clf.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
