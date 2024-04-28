import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import model_selection
from sklearn import metrics
from statsmodels.formula.api import ols


data_url = "http://lib.stat.cmu.edu/datasets/boston"
boston = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([boston.values[::2, :], boston.values[1::2, :2]])
target = boston.values[1::2, 2]
bos = pd.DataFrame(data)
feature_name = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B', 'LSTAT']
bos.columns = feature_name
bos['PRICE'] = target
bos.head()
model = ols("PRICE ~ CRIM + ZN + I(2**INDUS) + ZN:INDUS+ I(INDUS ** 2.0)", bos).fit()
# Print the summary
print((model.summary()))

print(pd.DataFrame(bos))
#lepszy model?
model = ols("PRICE ~ CRIM + I(INDUS) + ZN:INDUS + I(INDUS ** 2.0) + I(PTRATIO) + I(TAX**2) + I(NOX**2)", bos).fit()
# Print the summary
print((model.summary()))
