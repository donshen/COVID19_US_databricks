# Databricks notebook source
# MAGIC %md #A Study on Correlation of County-Wide US Demographics and Economy on COVID-19 Pandemic (Updated Aug 19th, 2020)

# COMMAND ----------

# MAGIC %md Let's visualize how bad the US situation is compared with the rest of the world, with a timeline.

# COMMAND ----------

#How corona virus infection grows world-wide?
from IPython.core.display import HTML
HTML('''<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1872044"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')

# COMMAND ----------

# MAGIC %md Import necessary packages and libraries for pyspark and visualization

# COMMAND ----------

# MAGIC %md ## Python libraries required that is not by default installed on dbfs: plotly, xgboost

# COMMAND ----------

import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.express as px
import plotly.graph_objs as go
#import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from urllib.request import urlopen
import sys
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
import seaborn as sn
from sklearn.decomposition import PCA

from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import re
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# COMMAND ----------

from pyspark.sql.types import StringType
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.types import IntegerType

# COMMAND ----------

# MAGIC %md Load and display COVID-19-US datasets (Updated till May 4th, 2020), as well as population, unemployment, and poverty datasets as spark dataframes

# COMMAND ----------

url_us_case = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
url_us_death = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
url_population = 'https://raw.githubusercontent.com/donshen/COVID19_US_databricks/master/data/PopulationEstimates.csv'
url_poverty = 'https://raw.githubusercontent.com/donshen/COVID19_US_databricks/master/data/PovertyEstimates.csv'
url_unemployment = 'https://raw.githubusercontent.com/donshen/COVID19_US_databricks/master/data/Unemployment.csv'

# COMMAND ----------

US_case = spark.createDataFrame(pd.read_csv(url_us_case))
US_death = spark.createDataFrame(pd.read_csv(url_us_death))
population_df = spark.createDataFrame(pd.read_csv(url_population, encoding= 'unicode_escape'))
poverty_df = spark.createDataFrame(pd.read_csv(url_poverty, encoding= 'unicode_escape'))
unemployment_df = spark.createDataFrame(pd.read_csv(url_unemployment,encoding= 'unicode_escape'))

# COMMAND ----------

# MAGIC %md Create a pyspark dataframe with key stats of COVID-19 in the US<br/>
# MAGIC Primary key: "FIPS" <br/>
# MAGIC (FIP 6-4, a five-digit Federal Information Processing Standards code which uniquely identified counties and county equivalents in the United States)  <br/>
# MAGIC Quality Analysis: fill nulls with 0, drop invalid rows, and clean some data that apparently doesn't make sense.

# COMMAND ----------

#Fill nulls with 0
US_case = US_case.fillna(0)
US_death = US_death.fillna(0)

#Extract useful columns (FIPS, County name, Population, Cases, Deaths)
df1 = US_case

keep_ColList = ['FIPS','Combined_Key',US_case.columns[-1]]
drop_ColList = list(set(US_case.columns)-set(keep_ColList))
df1 = df1.drop(*drop_ColList)
df1 = df1.withColumnRenamed(df1.columns[-1],"Cases")


df2 = US_death

keep_ColList = ['FIPS','Combined_Key','Population',US_case.columns[-1]]
drop_ColList = list(set(US_death.columns)-set(keep_ColList))
df2 = df2.drop(*drop_ColList)
df2 = df2.withColumnRenamed(df2.columns[-1],"Deaths")

#Join the reduced dataframes into a new dataframe called US_stat
US_stat = df1.join(df2,on = ['FIPS','Combined_Key'])
#Add another column named Fatality by computing the death rates
US_stat = US_stat.withColumn('InfectionRate',
                             F.format_number(US_stat.Cases/US_stat.Population,5).cast(DoubleType()))
US_stat = US_stat.withColumn('Fatality',
                             F.format_number(US_stat.Deaths/US_stat.Cases,4).cast(DoubleType()))
US_stat = US_stat.withColumnRenamed('Combined_Key','County')
US_stat = US_stat.fillna(0)
US_stat = US_stat.withColumn("FIPS",US_stat["FIPS"].cast(IntegerType()))
#Reformat FIPS to be the standard format with 5 digits
US_stat = US_stat.withColumn("FIPS",F.format_string("%05d","FIPS"))

# Although rows with deteriorated information has been removed previously, 
# some rows actually contain information that does not make sense, probably due to wrong data source
# such as population = 0, deaths > confirmed cases, etc.
US_stat = US_stat.filter((US_stat.Population != 0) & (US_stat.Cases >= US_stat.Deaths) & (US_stat.Population >= US_stat.Cases))

US_stat.show()

# COMMAND ----------

population_df = population_df.fillna(0)
#Reformat FIPS to be the standard format with 5 digits
population_df = population_df.withColumn("FIPS",population_df["FIPS"].cast(IntegerType()))
population_df = population_df.withColumn("FIPS",F.format_string("%05d","FIPS"))

removeComma = F.UserDefinedFunction(lambda x: re.sub(',','',str(x)), StringType())
removeDollarSign= F.UserDefinedFunction(lambda x: re.sub('\$','',x), StringType())

population_df = population_df.select(*[removeComma(column).alias(column) for column in population_df.columns])
population_df = population_df.select(*[removeDollarSign(column).alias(column) for column in population_df.columns])

#Only interested in the latest data, i.e. data of year 2018
features = [i for i in population_df.columns if '2018' in i]

#Only keep primary keys FIPS and other 2018 attributes
features.insert(0,'FIPS')
drop_ColList = list(set(population_df.columns)-set(features))
population_df = population_df.drop(*drop_ColList)

display(population_df)

# COMMAND ----------

#Fill nulls with 0
poverty_df = poverty_df.fillna(0)
#Reformat FIPS to be the standard format with 5 digits

poverty_df = poverty_df.withColumnRenamed('FIPStxt','FIPS')
poverty_df = poverty_df.withColumn("FIPS",poverty_df["FIPS"].cast(IntegerType()))
poverty_df = poverty_df.withColumn("FIPS",F.format_string("%05d","FIPS"))
poverty_df = poverty_df.select(*[removeComma(column).alias(column) for column in poverty_df.columns])
poverty_df = poverty_df.select(*[removeDollarSign(column).alias(column) for column in poverty_df.columns])


#Only interested in the latest data, i.e. data of year 2018
features = [i for i in poverty_df.columns if '2018' in i]

#Only keep primary keys FIPS and other 2018 attributes
features.insert(0,'FIPS')
drop_ColList = list(set(poverty_df.columns)-set(features))
poverty_df = poverty_df.drop(*drop_ColList)

display(poverty_df)

# COMMAND ----------

#Fill nulls with 0
unemployment_df = unemployment_df.fillna(0)
#Reformat FIPS to be the standard format with 5 digits
unemployment_df = unemployment_df.withColumn("FIPS",unemployment_df["FIPS"].cast(IntegerType()))
unemployment_df = unemployment_df.withColumn("FIPS",F.format_string("%05d","FIPS"))



unemployment_df = unemployment_df.select(*[removeComma(column).alias(column) for column in unemployment_df.columns])
unemployment_df = unemployment_df.select(*[removeDollarSign(column).alias(column) for column in unemployment_df.columns])

#Only interested in the latest data, i.e. data of year 2018
features = [i for i in unemployment_df.columns if '2018' in i]

#Only keep primary keys FIPS and other 2018 attributes
features.insert(0,'FIPS')
drop_ColList = list(set(unemployment_df.columns)-set(features))
unemployment_df = unemployment_df.drop(*drop_ColList)

display(unemployment_df)

# COMMAND ----------

# MAGIC %md Visualize total confirmed cases geographically and select the top ten counties with highest confirmed cases, the colorbar represents the value of log10(cases) of a specific county

# COMMAND ----------

df1 = US_stat.toPandas()
fig = px.choropleth(df1, geojson=counties, locations='FIPS', color=np.log10(df1.Cases),       
                    color_continuous_scale="viridis",
                    range_color=(0,np.max(np.log10(df1.Cases))),
                    scope="usa",
                   )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

US_stat.orderBy("Cases",ascending =False).limit(10).show()

# COMMAND ----------

# MAGIC %md Select the top ten counties with highest infection rate and visualize geographically

# COMMAND ----------

US_stat.orderBy("InfectionRate",ascending =False).limit(10).show()
fig = px.choropleth(df1, geojson=counties, locations='FIPS', color='InfectionRate',       
                    color_continuous_scale="viridis",
                    range_color=(0,np.max(df1.InfectionRate)/5),
                    scope="usa",  
                   )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

# COMMAND ----------

# MAGIC %md Select the top ten counties with highest death rate and visualize geographically

# COMMAND ----------

US_stat.orderBy("Fatality",ascending =False).limit(10).show()
fig = px.choropleth(df1, geojson=counties, locations='FIPS', color='Fatality',       
                    color_continuous_scale="viridis",
                    range_color=(0,0.1),
                    scope="usa",  
                   )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

# COMMAND ----------

# MAGIC %md Join population, poverty, unemployment dataframes and remove empty columns.
# MAGIC The data from the latest year (2018) is used.

# COMMAND ----------

US_stat = US_stat.join(population_df.
             join(poverty_df.
                  join(unemployment_df,on = 'FIPS',how='inner'),
                  on = 'FIPS',how='inner'),on = 'FIPS',how='inner')
columns_to_drop = ['POV04_2018','CI90LB04_2018','CI90UB04_2018','PCTPOV04_2018','CI90LB04P_2018','CI90UB04P_2018']
US_stat = US_stat.drop(*columns_to_drop)
US_stat.printSchema()
display(US_stat)

# COMMAND ----------

# MAGIC %md Select the couties with highest number of confirmed cases in descending order

# COMMAND ----------

US_case_top = US_case.orderBy(US_case.columns[-1],ascending =False).limit(8)
display(US_case_top)
US_case_top_pd = US_case_top.toPandas()

# COMMAND ----------

# MAGIC %md Select the couties with highest number of deaths in descending order

# COMMAND ----------

US_death_top = US_death.orderBy(US_death.columns[-1],ascending =False).limit(8)
display(US_death_top)
US_death_top_pd = US_death_top.toPandas()

# COMMAND ----------

n_days = US_case_top_pd.shape[1]-11
n_counties = sum(US_case_top_pd.iloc[:,-1]>0)
top_case_array = np.zeros((n_counties,n_days))
for i in range(n_counties):
  if US_case_top_pd.iloc[i,-1]>0:
    for j in range(n_days):
      top_case_array[i,j] = US_case_top_pd.iloc[i,j+11]


# COMMAND ----------

# MAGIC %md Use the logistic growth model to model the growth in total confirmed cases in US

# COMMAND ----------

from scipy.optimize import curve_fit
import matplotlib.dates as mdates
from matplotlib.dates import (YEARLY, DateFormatter,
                              rrulewrapper, RRuleLocator, drange)
import datetime as dt
from datetime import datetime
formatter = DateFormatter('%m/%d/%y')
  
def func(x, a, n, tau):
  return a /( 1+ n * np.exp(- x / tau))


dates = US_case_top_pd.columns[11:]

x_dates = [datetime.strptime(x, '%m/%d/%y') for x in dates]
xdata = np.linspace(1,n_days,n_days)

param_a = np.zeros(n_counties)
param_tau = np.zeros(n_counties)
nrow = 4
ncol = 2
fig, ax = plt.subplots(nrow,ncol,figsize=(18,12))

for i in range(n_counties):
  try:
    ydata = top_case_array[i,:]
    popt, pcov = curve_fit(func, xdata, ydata, absolute_sigma=True, p0 = np.array([70000,500000,6]), maxfev = 7000000)

    ax[int(np.floor(i/ncol)),i%ncol].plot_date(x_dates, ydata, 'b-', label='data')
    ax[int(np.floor(i/ncol)),i%ncol].plot_date(x_dates, func(xdata, *popt), 
                                            'g--', label='logistic: a=%5.3f, tau=%5.3f' % tuple([popt[0],popt[2]]))
    ax[int(np.floor(i/ncol)),i%ncol].xaxis.set_major_locator(mdates.MonthLocator())
    ax[int(np.floor(i/ncol)),i%ncol].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    ax[int(np.floor(i/ncol)),i%ncol].set_ylabel('Total Cases')
    ax[int(np.floor(i/ncol)),i%ncol].legend()
    ax[int(np.floor(i/ncol)),i%ncol].set_title(US_case_top_pd['Combined_Key'][i])
    param_a[i] = popt[0]
    param_tau[i] = popt[2]
    
    fig.autofmt_xdate()
  except:
    pass


display()

# COMMAND ----------

# MAGIC %md From the model, we can get the parameter a, which corresponds to the maximum number predicted by the model; and tau, which is essentially the inverse of the dayly growth rate r.

# COMMAND ----------

# MAGIC %md Visualize the ceilings of total confirmed cases, and the logistic growth rate

# COMMAND ----------

x = np.arange(len(US_case_top_pd['Combined_Key']))  # the label locations
width = 0.35  # the width of the bars

fig, ax1 = plt.subplots()
rects1 = ax1.bar(x - width/2, param_a, width, label='predicted maximum cases', color = 'blue',alpha = 0.6)
ax1.set_ylabel('Predicted maximum cases')
ax1.set_xticks(x)
ax1.set_ylim([0,5e5])
ax1.set_xticklabels(US_case_top_pd['Combined_Key'],rotation = 80)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
rects2 = ax2.bar(x + width/2, 1/param_tau, width, label='growth rate',color = 'crimson',alpha = 0.6)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylabel('Growth rate')

fig.tight_layout()
display()

# COMMAND ----------

n_days = US_death_top_pd.shape[1]-12
n_counties = sum(US_death_top_pd.iloc[:,-1]>0)
top_death_array = np.zeros((n_counties,n_days))
for i in range(n_counties):
  if US_death_top_pd.iloc[i,-1]>0:
    for j in range(n_days):
      top_death_array[i,j] = US_death_top_pd.iloc[i,j+12]

# COMMAND ----------

dates = US_death_top_pd.columns[12:]

x_dates = [datetime.strptime(x, '%m/%d/%y') for x in dates]
xdata = np.linspace(1,n_days,n_days)

param_a = np.zeros(n_counties)
param_tau = np.zeros(n_counties)
nrow = 4
ncol = 2
fig, ax = plt.subplots(nrow,ncol,figsize=(18,12))

for i in range(n_counties):
  try:
    ydata = top_death_array[i,:]
    popt, pcov = curve_fit(func, xdata, ydata, absolute_sigma=True, p0 = np.array([4500,600000,6]), maxfev = 1000000)

    ax[int(np.floor(i/ncol)),i%ncol].plot_date(x_dates, ydata, 'b-', label='data')
    ax[int(np.floor(i/ncol)),i%ncol].plot_date(x_dates, func(xdata, *popt), 
                                            'g--', label='logistic: a=%5.3f, tau=%5.3f' % tuple([popt[0],popt[2]]))
    ax[int(np.floor(i/ncol)),i%ncol].xaxis.set_major_locator(mdates.MonthLocator())
    ax[int(np.floor(i/ncol)),i%ncol].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    ax[int(np.floor(i/ncol)),i%ncol].set_ylabel('Total Deaths')
    ax[int(np.floor(i/ncol)),i%ncol].legend()
    ax[int(np.floor(i/ncol)),i%ncol].set_title(US_death_top_pd['Combined_Key'][i])
    param_a[i] = popt[0]
    param_tau[i] = popt[2]
    
    fig.autofmt_xdate()
  except:
    pass


display()

# COMMAND ----------

x = np.arange(len(US_death_top_pd['Combined_Key']))  # the label locations
width = 0.35  # the width of the bars

fig, ax1 = plt.subplots()
rects1 = ax1.bar(x - width/2, param_a, width, label='predicted maximum deaths', color = 'blue',alpha = 0.6)
ax1.set_ylabel('Predicted maximum deaths')
ax1.set_xticks(x)
ax1.set_xticklabels(US_death_top_pd['Combined_Key'],rotation = 80)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
rects2 = ax2.bar(x + width/2, 1/param_tau, width, label='growth rate',color = 'crimson',alpha = 0.6)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylabel('Growth rate')

fig.tight_layout()
display()

# COMMAND ----------

# MAGIC %md  We now want to apply this to all the counties that has non-zero confirmed COVID-19 cases.

# COMMAND ----------

# MAGIC %md # Implement Principle Component Regression 
# MAGIC Perform PCR using features from population, unemployment, and poverty Data

# COMMAND ----------

US_stat_pd = US_stat.toPandas()
US_stat_pd.fillna(0)
US_stat_pd = US_stat_pd.dropna(axis = 0)
US_stat_pd.head()

# COMMAND ----------

#Find null/None values in the converted dataframe
na_result = US_stat_pd.eq('None')

seriesObj = na_result.any()
columnNames = list(seriesObj[seriesObj == True].index)
for col in columnNames:
    rows = list(na_result[col][na_result[col] == True].index)

#Since pyspark does not support drop rows by index, pandas was used here
US_stat_pd = US_stat_pd.drop(rows)

# COMMAND ----------

# MAGIC %md Will use the "POP_ESTIMATE_2018" instead for consistency, and the values are pretty close to the ones from COVID-19 datasets.

# COMMAND ----------

# MAGIC %md 40+ features from population, poverty, and unemployment datasets (X), and four responses: confirmed cases (Ycase), confirmed deaths (Ydeath), infection rates (Yinfection), and death rate (Yfatality)

# COMMAND ----------

X = US_stat_pd.iloc[:,7:].astype('double')

Y_death = US_stat_pd['Deaths'].astype('double')
Y_case = US_stat_pd['Cases'].astype('double')
Y_infection = US_stat_pd['InfectionRate'].astype('double')
Y_fatality = US_stat_pd['Fatality'].astype('double')

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X[X.columns] = pd.DataFrame(scaler.fit_transform(X[X.columns]))


# COMMAND ----------

# Split lateset confirmed cases into training (80%) and test(20%) sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_case, test_size=0.2, random_state= 0)

#PCA on training data
pca_train = PCA(n_components=X_train.shape[1])
pc_train = pca_train.fit_transform(np.nan_to_num(X_train))


# COMMAND ----------

from sklearn.linear_model import LinearRegression
Rsq = np.zeros(X_train.shape[1])

for i in range(X_train.shape[1]):
  X_pc_train = pc_train[:,:i+1]
  reg = LinearRegression().fit(X_pc_train[Y_train>0,:], np.log10(Y_train[Y_train>0]))
  Rsq[i] = reg.score(X_pc_train[Y_train>0,:], np.log10(Y_train[Y_train>0]))

fig, ax1 = plt.subplots(figsize = (5,3))
ax1.bar(range(X_train.shape[1]),pca_train.explained_variance_ratio_,color='blue')
ax1.set_ylabel('Explained Variance')
ax1.set_xlabel('N_PC')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(range(X_train.shape[1]),Rsq,color = 'crimson')
ax2.set_ylabel('R_square')
ax2.tick_params(axis='y', labelcolor='crimson')
display()

# COMMAND ----------

# MAGIC %md Perform Principle Component Regression on Test sets, choosing 12 principle components according to the previous plot

# COMMAND ----------

loadings = pca_train.components_.T * np.sqrt(pca_train.explained_variance_)

X_pc_test = np.dot(np.asarray(X_test),np.asarray(loadings))

n_pc = 15
reg = LinearRegression().fit(X_pc_test[Y_test>0,:n_pc+1], np.log10(Y_test[Y_test>0]))
reg.score(X_pc_test[Y_test>0,:n_pc+1], np.log10(Y_test[Y_test>0]))
plt.scatter(reg.predict(X_pc_test[Y_test>0,:n_pc+1]),np.log10(Y_test[Y_test>0]), color='crimson',alpha=0.6)
plt.xlabel("log10(#Cases)_pred")
plt.ylabel("log10(#Cases)_test")
plt.xlim([0,5])
plt.ylim([0,5])
plt.gca().set_aspect('equal', adjustable='box')
display()

# COMMAND ----------

# MAGIC %md # Implement XGBoost
# MAGIC predicting COVID-19 stats from population, unemployment, and poverty Data

# COMMAND ----------

# MAGIC %md Classification on whether a county has high fatality (greater or equal than 1% of infected are reported dead)

# COMMAND ----------

#Use XGBoost to classify counties with fatality rate > 0.1 (y = 1) and otherwise (y = 0)
pos = [Y_fatality >= 0.01]
neg = [Y_fatality < 0.01]

Y_fatality_response = np.array(Y_fatality)
Y_fatality_response[pos] = 1
Y_fatality_response[neg] = 0

# First XGBoost model 
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X,Y_fatality_response, test_size=test_size, random_state=seed)

# COMMAND ----------

# MAGIC %md Hyper-parameter tuning for XGBoost using RandomizedSearchCV

# COMMAND ----------

from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats

clf = XGBClassifier(n_estimators=500)

param_dist = {'max_depth': np.arange(5, 10),
              'colsample_bytree': np.arange(0.6,1,0.1),
              'subsample': np.arange(0.6,1,0.1),
              'eta': np.logspace(-2, 0, 10)
             }

n_iter_search = 20

random_search = RandomizedSearchCV(
    clf,
    param_distributions=param_dist,
    n_iter=n_iter_search,
    cv=5,
    scoring='accuracy'
)

random_search.fit(
    X_train,
    y_train,
    eval_metric='logloss',
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=50,
    early_stopping_rounds = 20
)

# COMMAND ----------

# best estimator
print(random_search.best_estimator_)

# COMMAND ----------

# fit model with training data using the optimized parameters
model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=0.7999999999999999,
       eta=0.027825594022071243, gamma=0, gpu_id=-1,
       importance_type='gain', interaction_constraints='',
       learning_rate=0.0278255939, max_delta_step=0, max_depth=6,
       min_child_weight=1, missing=None, monotone_constraints='()',
       n_estimators=500, n_jobs=0, num_parallel_tree=1,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, subsample=0.6,
       tree_method='exact', validate_parameters=1, verbosity=None)
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# COMMAND ----------

# MAGIC %md Calculate precision, recall, and F1 score of the model

# COMMAND ----------

from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
f1 = f1_score(y_test, y_pred)
precision, recall, _ = precision_recall_curve(y_test, y_pred)
precision = precision[1]
recall = recall[1]
print("Precison: %.2f\nRecall: %.2f\nF1 score:  %.2f" % (precision, recall, f1))

# COMMAND ----------

# MAGIC %md Feaure importance

# COMMAND ----------

import xgboost as xgb
xgb.plot_importance(model)
plt.rcParams['figure.figsize'] = [15,15]
display()

# COMMAND ----------

temp  = pd.concat([US_stat_pd,pd.DataFrame(Y_fatality_response,columns = ['Fatality>0.01'])],axis =1)
fig = px.choropleth(temp, geojson=counties, locations='FIPS', color='Fatality>0.01',       
                    color_continuous_scale="viridis",
                    range_color=(0,1),
                    scope="usa",  
                   )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# COMMAND ----------

# MAGIC %md Classification on whether a county is considered as highly infected (> 1% of population got infected to-date)

# COMMAND ----------

# Tain-test split
pos = Y_infection>=0.01
Y_infection_response = np.zeros(len(Y_infection))
Y_infection_response[pos] = 1


seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y_infection_response, test_size=test_size, random_state=seed)

# COMMAND ----------

clf = XGBClassifier(n_estimators=500)

param_dist = {'max_depth': np.arange(5, 10),
              'colsample_bytree': np.arange(0.6,1,0.1),
              'subsample': np.arange(0.6,1,0.1),
              'eta': np.logspace(-2, 0, 10)
             }

n_iter_search = 20

random_search = RandomizedSearchCV(
    clf,
    param_distributions=param_dist,
    n_iter=n_iter_search,
    cv=5,
    scoring='accuracy'
)

random_search.fit(
    X_train,
    y_train,
    eval_metric='logloss',
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=50,
    early_stopping_rounds = 20
)

# COMMAND ----------

# best estimator
print(random_search.best_estimator_)

# COMMAND ----------

# fit the XGBoost model with optimized parameters 
model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=0.7, eta=0.01, gamma=0,
       gpu_id=-1, importance_type='gain', interaction_constraints='',
       learning_rate=0.00999999978, max_delta_step=0, max_depth=8,
       min_child_weight=1, missing=None, monotone_constraints='()',
       n_estimators=500, n_jobs=0, num_parallel_tree=1,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, subsample=0.7,
       tree_method='exact', validate_parameters=1, verbosity=None)
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# COMMAND ----------

f1 = f1_score(y_test, y_pred)
precision, recall, _ = precision_recall_curve(y_test, y_pred)
precision = precision[1]
recall = recall[1]
print("Precison: %.2f\nRecall: %.2f\nF1 score:  %.2f" % (precision, recall, f1))

# COMMAND ----------

temp  = pd.concat([US_stat_pd,pd.DataFrame(Y_infection_response,columns = ['Infection>0.005'])],axis =1)
fig = px.choropleth(temp, geojson=counties, locations='FIPS', color='Infection>0.005',       
                    color_continuous_scale='viridis',
                    range_color=(0,1),
                    scope="usa",  
                   )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.update(layout_coloraxis_showscale=False)
fig.show()

# COMMAND ----------

# MAGIC %md Feature importance in classifying high infection area

# COMMAND ----------

import xgboost as xgb
xgb.plot_importance(model)
plt.rcParams['figure.figsize'] = [15,15]
display()

# COMMAND ----------

# MAGIC %md ## Forward Stepwise Feature Selection 

# COMMAND ----------

# MAGIC %md Usually a best linear model could be selected out of all combinations of the predictors using best subset selection algorithm. However, given the tremendous amount of predictors here, this is computationally expensive. Therefore, a similar but less costly approach: forward stepwise selection is used here.

# COMMAND ----------

# MAGIC %md First, we perform forward stepwise feature selection on X when the response is Y_case (total confirmed cases)

# COMMAND ----------

import itertools
import time
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error



# COMMAND ----------

def processSubset(feature_set):
    # Fit model on feature_set and calculate RSS
    model = sm.OLS(y,X[list(feature_set)])
    regr = model.fit()
    RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()
    return {"model":regr, "RSS":RSS}
  
def getBest(k):
    
    tic = time.time()
    
    results = []
    
    for combo in itertools.combinations(X.columns, k):
        results.append(processSubset(combo))
    
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    
    toc = time.time()
    print("Processed", models.shape[0], "models on", k, "predictors in", (toc-tic), "seconds.")
    
    # Return the best model, along with some other useful information about the model
    return best_model

# COMMAND ----------

def forward(predictors):

    # Pull out predictors we still need to process
    remaining_predictors = [p for p in X.columns if p not in predictors]
    
    tic = time.time()
    
    results = []
    
    for p in remaining_predictors:
        results.append(processSubset(predictors+[p]))
    
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors)+1, "predictors in", (toc-tic), "seconds.")
    
    # Return the best model, along with some other useful information about the model
    return best_model

# COMMAND ----------

X = US_stat_pd.iloc[:,7:].astype('double')

Y_death = US_stat_pd['Deaths'].astype('double')
Y_case = US_stat_pd['Cases'].astype('double')

# COMMAND ----------

# Normalize X using min&max scaling
# Transform the response by log10(1+Y)
X=(X-X.min())/(X.max()-X.min())
y = np.log10(Y_case+1)

# COMMAND ----------

# Transform the response by log10(1+Y)
y = np.log10(Y_case+1)

models_fwd = pd.DataFrame(columns=["RSS", "model"])

tic = time.time()
predictors = []

for i in range(1,len(X.columns)+1):    
    models_fwd.loc[i] = forward(predictors)
    predictors = models_fwd.loc[i]["model"].model.exog_names

toc = time.time()
print("Total elapsed time:", (toc-tic), "seconds.")

# COMMAND ----------

plt.figure(figsize=(12,8))
plt.rcParams.update({'font.size': 14, 'lines.markersize': 8})

# Set up a 2x2 grid so we can look at 4 plots at once
plt.subplot(2, 2, 1)

# We will now plot a red dot to indicate the model with the largest adjusted R^2 statistic.
# The argmax() function can be used to identify the location of the maximum point of a vector
plt.plot(models_fwd["RSS"])
plt.xlabel('# Predictors')
plt.ylabel('RSS')

# We will now plot a red dot to indicate the model with the largest adjusted R^2 statistic.
# The argmax() function can be used to identify the location of the maximum point of a vector

rsquared_adj = models_fwd.apply(lambda row: row[1].rsquared_adj, axis=1)

plt.subplot(2, 2, 2)
plt.plot(rsquared_adj)
plt.plot(rsquared_adj.argmax(), rsquared_adj.max(), "or")
plt.xlabel('# Predictors')
plt.ylabel('adjusted Rsquared')

# We'll do the same for AIC and BIC, this time looking for the models with the SMALLEST statistic
aic = models_fwd.apply(lambda row: row[1].aic, axis=1)

plt.subplot(2, 2, 3)
plt.plot(aic)
plt.plot(aic.argmin(), aic.min(), "or")
plt.xlabel('# Predictors')
plt.ylabel('AIC')

bic = models_fwd.apply(lambda row: row[1].bic, axis=1)

plt.subplot(2, 2, 4)
plt.plot(bic)
plt.plot(bic.argmin(), bic.min(), "or")
plt.xlabel('# Predictors')
plt.ylabel('BIC')

# COMMAND ----------

print("-----------------")
print("Foward Selection:")
print("-----------------")
print(models_fwd.loc[bic.argmin(), "model"].summary())

# COMMAND ----------

pred_val = models_fwd.loc[bic.argmin(), "model"].fittedvalues.copy()
true_val = y.values.copy()
residual = true_val - pred_val
fig, ax = plt.subplots(figsize=(8,4))
ax.scatter(true_val,residual,alpha = 0.6)
ax.set_xlabel('log10(1+Y_case)')
ax.set_ylabel('Residual')

# COMMAND ----------

# MAGIC %md Second, we perform forward stepwise feature selection on X when the response is Y_death (total confirmed deaths)

# COMMAND ----------

# Transform the response by log10(1+Y)
y = np.log10(Y_death+1)

models_fwd = pd.DataFrame(columns=["RSS", "model"])

tic = time.time()
predictors = []

for i in range(1,len(X.columns)+1):    
    models_fwd.loc[i] = forward(predictors)
    predictors = models_fwd.loc[i]["model"].model.exog_names

toc = time.time()
print("Total elapsed time:", (toc-tic), "seconds.")

# COMMAND ----------

plt.figure(figsize=(12,8))
plt.rcParams.update({'font.size': 14, 'lines.markersize': 8})

# Set up a 2x2 grid so we can look at 4 plots at once
plt.subplot(2, 2, 1)

# We will now plot a red dot to indicate the model with the largest adjusted R^2 statistic.
# The argmax() function can be used to identify the location of the maximum point of a vector
plt.plot(models_fwd["RSS"])
plt.xlabel('# Predictors')
plt.ylabel('RSS')

# We will now plot a red dot to indicate the model with the largest adjusted R^2 statistic.
# The argmax() function can be used to identify the location of the maximum point of a vector

rsquared_adj = models_fwd.apply(lambda row: row[1].rsquared_adj, axis=1)

plt.subplot(2, 2, 2)
plt.plot(rsquared_adj)
plt.plot(rsquared_adj.argmax(), rsquared_adj.max(), "or")
plt.xlabel('# Predictors')
plt.ylabel('adjusted Rsquared')

# We'll do the same for AIC and BIC, this time looking for the models with the SMALLEST statistic
aic = models_fwd.apply(lambda row: row[1].aic, axis=1)

plt.subplot(2, 2, 3)
plt.plot(aic)
plt.plot(aic.argmin(), aic.min(), "or")
plt.xlabel('# Predictors')
plt.ylabel('AIC')

bic = models_fwd.apply(lambda row: row[1].bic, axis=1)

plt.subplot(2, 2, 4)
plt.plot(bic)
plt.plot(bic.argmin(), bic.min(), "or")
plt.xlabel('# Predictors')
plt.ylabel('BIC')

# COMMAND ----------

print("-----------------")
print("Foward Selection:")
print("-----------------")
print(models_fwd.loc[bic.argmin(), "model"].summary())

# COMMAND ----------

pred_val = models_fwd.loc[bic.argmin(), "model"].fittedvalues.copy()
true_val = y.values.copy()
residual = true_val - pred_val
fig, ax = plt.subplots(figsize=(8,4))
ax.scatter(true_val,residual,alpha = 0.6)
ax.set_xlabel('log10(1+Y_death)')
ax.set_ylabel('Residual')
