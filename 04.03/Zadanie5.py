from pandas import DataFrame
import pandas as pd

df = pd.read_csv('airports.csv')

# 1

print(df.tail(12)['iso_country'], "\n"*5)

# 2
print(df.loc[1], "\n"*3)
print(df.iloc[1], "\n"*5)

# 3
print(df[df.iso_country == 'PL'])

# 4
print(df[df.name != df.municipality])

# 5
print(df.elevation_ft)
df.elevation_ft = df.elevation_ft * 30.48 / 100
print(df.elevation_ft)

# 6
a = df.iso_country.duplicated(keep=False)
for i in df.iso_country.unique():
    if df['iso_country'].value_counts().get(i, 0) == 1:
        print(i)

