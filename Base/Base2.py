import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
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

dataset.info()

# from sklearn.model_selection import train_test_split



# X = dataset.drop(columns=existing_columns_to_drop)
# y = dataset["Label"]

# Check if columns exist before dropping
columns_drop = ["Label", "year", "month", "day", "hour"]
existing_columns_to_drop = [col for col in columns_drop if col in dataset.columns]

# Split the data based on the month and year
train_data = dataset[~((dataset['year'] == 2019) & (dataset['month'] == 10))]
test_data = dataset[(dataset['year'] == 2019) & (dataset['month'] == 10)]

# Separate features and labels
X_train = train_data.drop(columns=existing_columns_to_drop)
y_train = train_data['Label']
X_test = test_data.drop(columns=existing_columns_to_drop)
y_test = test_data['Label']

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

print("Training set label distribution:")
print(y_train.value_counts())
print("Test set label distribution:")
print(y_test.value_counts())


# Print label distribution in the training and test sets
train_label_distribution = y_train.value_counts(normalize=True) * 100
test_label_distribution = y_test.value_counts(normalize=True) * 100

print("Training set label distribution (percentage):")
print(train_label_distribution)
print("Test set label distribution (percentage):")
print(test_label_distribution)


baserf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)

baserf.fit(X_train, y_train)

y_pred = baserf.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))