import pandas as pd
import numpy as np
df = pd.read_csv('https://github.com/Ulvi-Movs/titanic/raw/main/train.csv')
print(df)
df = df.dropna()
print(df)