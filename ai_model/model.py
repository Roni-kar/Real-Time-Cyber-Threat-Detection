import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("1. Loading the NSL-KDD dataset...")
df = pd.read_csv('ai_model/KDDTrain+.txt', header=None) 

print("2. Formatting the data...")
X = df.iloc[:, :-2] 
y = df.iloc[:, -2]
X = pd.get_dummies(X)

X.columns = X.columns.astype(str)

print("3. Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("4. Training the Random Forest AI on the RTX 3050...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

print("5. Testing Model Accuracy...")
predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy Achieved: {accuracy * 100:.2f}%")

print("6. Exporting the trained AI model...")
joblib.dump(clf, 'ai_model/cyber_threat_model.pkl')
print("Model saved! Ready to push to GitHub.")