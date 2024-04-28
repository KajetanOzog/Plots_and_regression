import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import statsmodels.formula.api as smf

df_adv = pd.read_csv('https://raw.githubusercontent.com/przem85/bootcamp/master/statistics/Advertising.csv', index_col=0)
df_adv.head()

print(pd.DataFrame(df_adv))

est = smf.ols(formula='sales ~ I(newspaper)*I(TV)*I(radio)', data=df_adv).fit()
print((est.summary2()))

# est = smf.ols(formula='sales ~ I(newspaper)*I(TV)*I(radio)', data=df_adv).fit()

est = smf.ols(formula='sales ~ I(TV) + I(radio) + I(TV**2) +  I(TV):I(radio)  + I(TV**3) + I(TV**4) + I(TV**5)'
                      ' + I(TV**6) ', data=df_adv).fit()
print((est.summary2()))

est = smf.ols(formula='sales ~ I(newspaper)+I(TV):I(radio)+np.log(radio+1)', data=df_adv).fit()
print((est.summary2()))


est = smf.ols(formula='sales ~ I(radio) + I(TV**2) +  I(TV):I(radio)  + I(TV**3) + I(TV**4) + I(TV**5)'
                      ' + I(TV**6)', data=df_adv).fit() #chyba lepszy model
print((est.summary2()))