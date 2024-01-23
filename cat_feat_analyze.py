import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze():
    titanic_train_df = pd.read_csv("train.csv")

    print(titanic_train_df.head())

    cont_feat = ['PassengerId', 'Pclass', 'Name', 'Ticket', 'Age', 'SibSp', 'SibSp', 'Fare', 'Parch']
    titanic_train_df.drop(cont_feat, axis=1, inplace=True)
    print(titanic_train_df.head())

    print(titanic_train_df.info())

    titanic_train_df['Cabin_indicator'] = np.where(titanic_train_df['Cabin'].isnull(), 0, 1)
    print(titanic_train_df.head())

    for feat, col in enumerate(['Cabin_indicator', 'Sex', 'Embarked']):
        plt.figure(feat)
        sns.catplot(x=col, y='Survived', data=titanic_train_df, kind='point', aspect=2,)
    plt.show()

    print(titanic_train_df.pivot_table('Survived', index='Sex', columns='Embarked', aggfunc='count'))

    print(titanic_train_df.pivot_table('Survived', index='Cabin_indicator', columns='Embarked', aggfunc='count'))

    gender = {'male': 0, 'female': 1}
    titanic_train_df['Sex'] = titanic_train_df['Sex'].map(gender)
    print(titanic_train_df.head())
    titanic_train_df.drop(['Cabin', 'Embarked'], axis=1, inplace=True)
    print(titanic_train_df)

    return titanic_train_df

