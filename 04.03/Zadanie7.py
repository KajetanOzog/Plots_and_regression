import pandas as pd
import numpy as np
df = pd.read_csv('https://github.com/Ulvi-Movs/titanic/raw/main/train.csv')
df = df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

df["HasCabin"] = np.where(df.Cabin.isnull(), 0, 1)

print(df)

