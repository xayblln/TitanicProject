import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

test_features = pd.read_csv("test_features.csv")
test_labels = pd.read_csv("test_labels.csv")
train_features = pd.read_csv("train_features.csv")
train_labels = pd.read_csv("train_labels.csv")

rf = RandomForestClassifier(n_estimators=100, max_depth=12)
rf.fit(train_features, train_labels.values.ravel())

y_pred = rf.predict(test_features)

print('Accuracy: ', str(accuracy_score(test_labels, y_pred)))
print('Precision: ', str(precision_score(test_labels, y_pred)))
print('Recall: ', str(recall_score(test_labels, y_pred)))