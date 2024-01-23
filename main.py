import cat_feat_analyze
import cont_feat_analyze
import pandas as pd
import numpy as np

import data_save

if __name__ == '__main__':
    # cont_feat_analyze.analyse()
    # cat_feat_analyze.analyze()

    titanic_df = pd.read_csv("train.csv")

    titanic_df['Family'] = titanic_df['SibSp'] + titanic_df['Parch']
    titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)
    titanic_df.drop(['Parch', 'SibSp'], axis=1, inplace=True)

    titanic_df['Cabin_indicator'] = np.where(titanic_df['Cabin'].isnull(), 0, 1)
    gender = {'male': 0, 'female': 1}
    titanic_df['Sex'] = titanic_df['Sex'].map(gender)
    titanic_df.drop(['Cabin', 'Embarked', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

    data_save.saving(titanic_df)

    print(titanic_df.head())
