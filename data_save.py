from sklearn.model_selection import train_test_split

def saving(titanic_df):
    features = titanic_df.drop(['Survived'], axis=1)
    labels = titanic_df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

    X_train.to_csv("train_features.csv", index=False)
    X_val.to_csv("val_features.csv", index=False)
    X_test.to_csv("test_features.csv", index=False)

    y_train.to_csv("train_labels.csv", index=False)
    y_val.to_csv("val_labels.csv", index=False)
    y_test.to_csv("test_labels.csv", index=False)
