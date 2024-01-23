import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyse():
    titanic_train_df = pd.read_csv("train.csv")
    print(titanic_train_df.head())
    print(titanic_train_df.info())

    cat_feat = ['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
    titanic_train_df.drop(cat_feat, axis=1, inplace=True)

    print(titanic_train_df.head())

    print(titanic_train_df.describe())

    print(titanic_train_df.groupby('Survived').mean())

    print(titanic_train_df.groupby(titanic_train_df['Age'].isnull()).mean())

    for feat in ['Age', 'Fare']:
        died = list(titanic_train_df[titanic_train_df['Survived'] == 0][feat].dropna())
        survived = list(titanic_train_df[titanic_train_df['Survived'] == 1][feat].dropna())
        xmin = min(min(died), min(survived))
        xmax = max(max(died), max(survived))
        width = (xmax - xmin) / 40
        sns.histplot(died, color='r', kde=False, bins=np.arange(xmin, xmax, width))
        sns.histplot(survived, color='g', kde=False, bins=np.arange(xmin, xmax, width))
        plt.legend(['Died', 'Survived'])
        plt.title('Overlaid histogram for {}'.format(feat))
        plt.show()

    for feat, col in enumerate(['Pclass', 'SibSp', 'Parch']):
        plt.figure(feat)
        sns.catplot(x=col, y='Survived', data=titanic_train_df, kind='point', aspect=2, )
    plt.show()

    titanic_train_df['Family'] = titanic_train_df['SibSp'] + titanic_train_df['Parch']
    sns.catplot(x='Family', y='Survived', data=titanic_train_df, kind='point', aspect=2, )
    plt.show()

    titanic_train_df['Age'].fillna(titanic_train_df['Age'].mean(), inplace=True)
    print(titanic_train_df.isnull().sum())
    print(titanic_train_df)

    titanic_train_df.drop(['Parch', 'SibSp'], axis=1, inplace=True)
    print(titanic_train_df.head())

    return titanic_train_df
