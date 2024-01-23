import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score

def print_res(results):
    print('Optimal Hyperparams: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']

    for mean, params in zip(means, results.cv_results_['params']):
        print('Mean: {} Hyperparams: {}'.format(round(mean, 3), params))

def cross_val():
    train_features = pd.read_csv("train_features.csv")
    train_labels = pd.read_csv("train_labels.csv")

    hp = {
        'n_estimators': [5, 25, 50, 100],
        'max_depth': [2, 12, 24, None]
    }

    rf = RandomForestClassifier()

    cross_val_model = GridSearchCV(rf, hp, cv=5)
    cross_val_model.fit(train_features, train_labels.values.ravel())

    print_res(cross_val_model)

    rf1 = RandomForestClassifier(n_estimators=100, max_depth=12)
    rf1.fit(train_features, train_labels.values.ravel())

    rf2 = RandomForestClassifier(n_estimators=50, max_depth=12)
    rf2.fit(train_features, train_labels.values.ravel())

    rf3 = RandomForestClassifier(n_estimators=25, max_depth=12)
    rf3.fit(train_features, train_labels.values.ravel())

    val_features = pd.read_csv("val_features.csv")
    val_labels = pd.read_csv("val_labels.csv")

    for mdl in [rf1, rf2, rf3]:
        y_pred = mdl.predict(val_features)

        print("Max Depth: {} Estimators: {}\n".format(mdl.max_depth, mdl.n_estimators))
        print('Accuracy: ', str(accuracy_score(val_labels, y_pred)))
        print('Precision: ', str(precision_score(val_labels, y_pred)))
        print('Recall: ', str(recall_score(val_labels, y_pred)))
        print()


cross_val()
