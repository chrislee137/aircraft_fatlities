import pandas as pd
import psycopg2 as pg
import numpy as np
import sys, argparse, csv
from sklearn.model_selection import train_test_split   
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import statsmodels.formula.api as smf
import statsmodels.api as sm
%matplotlib inline
plt.style.use('ggplot')
from pandas.plotting import scatter_matrix
import pylab as py

def crashes():
    crashes = pd.read_csv("../data/aircraft_info.csv")
if __name__ == '__main__':
    crashed= crashes()
crashes = pd.read_csv(r"C:\Users\chris\anaconda3\cohort_06\daimil10\exams\aircraft_fatlities\data\aircraft_info.csv")
crashed['Date'] = pd.to_datetime(crashed['Date'], dayfirst=False)
crashed['Date'] = crashed['Date'].dt.strftime('%Y-%m')
new_dates= crashed[(crashed['Date'] > '2015-01')]

X= new_dates['Aboard Passangers']
y= new_dates['Fatalities Passangers']
X = sm.add_constant(X)
simple_model = sm.OLS(y,X).fit()
simple_predictions = simple_model.predict(X)

print_simple_table = simple_model.summary()
print(print_simple_table)

sm.graphics.plot_fit(simple_model,'Aboard Passangers')

X=new_dates['Aboard Passangers'].values
Y=new_dates['Fatalities Passangers'].values
from scipy.stats import linregress
m ,b, *_ = stats.linregress(X,Y)

f = new_dates.sort_values(('Aboard Passangers'), axis = 0, ascending = False)
ax = f.plot(kind='bar', title ="Flight Survivability", figsize=(10,5),legend=True, fontsize=8)

ax.set_xlabel("Crash Input ID",fontsize=8)
ax.set_ylabel("Passagngers",fontsize=8)

if __name__ == '__main__':
    crashes, newd_dates = read_data()