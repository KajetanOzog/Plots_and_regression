import pandas as pd

df = pd.read_csv('https://github.com/Ulvi-Movs/titanic/raw/main/train.csv')
df = df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
print(df.columns)


