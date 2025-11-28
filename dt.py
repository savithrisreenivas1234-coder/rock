import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap

# Corrected path
data = pd.read_csv(r"C:\Users\SRINIVAS\OneDrive\Documents\csv\User_Data.csv")

# Features and Target
x = data.iloc[:, [2, 3]].values
y = data.iloc[:, 4].values

# Corrected path
data = pd.read_csv(r"C:\Users\SRINIVAS\OneDrive\Documents\csv\User_Data.csv")

# Features and Target
x = data.iloc[:, [2, 3]].values
y = data.iloc[:, 4].values

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Standard Scaling
Scaler = StandardScaler()
x_train = Scaler.fit_transform(x_train)
x_test = Scaler.transform(x_test)

# Decision Tree Classifier
classifier = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=0)
classifier.fit(x_train, y_train)

# Predictions
y_pred = classifier.predict(x_test)

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualization Boundary
x1, x2 = np.meshgrid(
    np.arange(x_train[:, 0].min() - 1, x_train[:, 0].max() + 1, 0.01),
    np.arange(x_train[:, 1].min() - 1, x_train[:, 1].max() + 1, 0.01)
)

plt.contourf(
    x1, x2,
    classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
    alpha=0.75,
    cmap=ListedColormap(["red", "green"])
)

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

# Training points
for i, j in enumerate(np.unique(y_train)):
    plt.scatter(
        x_train[y_train == j, 0],
        x_train[y_train == j, 1],
        color=ListedColormap(["red", "green"])(i),
        label=j
    )

plt.title("Decision Tree Classifier")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.legend()
plt.show()
