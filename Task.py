import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss

data = pd.read_csv(r'C:\Users\Hripsime\Desktop\DataArt\ML_Station_3_Project_Neural_Network_Problem_Solver\your_dataset.csv')
print(data.columns)

if data.isnull().sum().any():
    data = data.fillna(data.mean())

target_column = 'accepted for the interview'
if target_column not in data.columns:
    raise KeyError(f"'{target_column}' not found in the dataset")

X = data.drop(target_column, axis=1)
y = data[target_column]

# Label Encoding for categorical features
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

model.fit(X_train, y_train)

train_loss = log_loss(y_train, model.predict_proba(X_train))
val_loss = log_loss(y_test, model.predict_proba(X_test))

train_accuracy = accuracy_score(y_train, model.predict(X_train))
val_accuracy = accuracy_score(y_test, model.predict(X_test))

plt.plot(model.loss_curve_)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()

print(f"Train Loss: {train_loss}")
print(f"Validation Loss: {val_loss}")
print(f"Train Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
