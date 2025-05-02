import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Citire dataset
dataset_path = 'Skin_database.txt'
df = pd.read_csv(dataset_path, sep='\t', header=None, names=['B', 'G', 'R', 'Class'], engine='python')

# Separare caracteristici și ținte
X = df[['B', 'G', 'R']]
y = df['Class']

# Normalizare
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Împărțire în seturi de antrenament și test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Inițializare și antrenare MLP
mlp_model = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', solver='adam', max_iter=1000, random_state=42)
mlp_model.fit(X_train, y_train)

# Predicții
y_pred = mlp_model.predict(X_test)

# Evaluare
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Matrice de confuzie
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - MLP')
plt.show()
