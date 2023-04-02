#!/usr/bin/env python
# coding: utf-8

# ## EDA & Data Visualization

# Problem Statement:  We are tasked with creating an automated system to estimate the annual medical charges for new customers, using information such as their age, sex, BMI, children, smoking habits and region of residence. Estimates from the system will be used to determine the annual insurance premium offered to the customer.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('insurance.csv')
df.head(10)


# In[3]:


df.info()


# In[4]:


df.isna().sum()


# As we can see there are no null values. However, let's check for duplicates.

# In[5]:


df.duplicated().sum()


# In[6]:


df[df.duplicated(keep=False)]


# In[7]:


df = df.drop_duplicates()
df.duplicated().sum()


# We found a single duplicate which is dropped now.

# In[8]:


df.describe().T


# Now let us distinguish the Body Mass Indexes into the following categories:

# In[9]:


def bmi_to_cat(bmi: float) -> str:
    if bmi < 18.5: return 'Underweight'
    elif 18.5 <= bmi < 25: return 'Normal Weight'
    elif 25 <= bmi < 30: return 'Overweight'
    elif 30 <= bmi < 35: return 'Obese'
    elif 35 <= bmi < 40: return 'Severely Obese'
    elif 40 <= bmi: return 'Morbidly Obese'
    
df.insert(3, 'bmi_cat', df['bmi'].apply(bmi_to_cat))
df.head()


# In[10]:


fig = px.scatter(df, x="bmi", y="age", color="bmi_cat")
fig.show()


# Let's dive deep into visualizing different categorical variables with respect to charges.

# In[11]:


pd.crosstab(df.sex, df.smoker, margins = True)


# In[12]:


fig = px.box(df, x="sex", y="charges", color="smoker")
fig.show()


# Observations:
# For most customers, the annual medical charges are about 10,000 dollars. But there are few outliers who have higher medical expenses, possibly due to some other reasons.
# However, there is a significant difference in medical expenses between smokers and non-smokers. While the median for non-smokers is 7300 dollars, the median for smokers is close to 32,000 dollars.

# In[13]:


pd.crosstab([df.sex, df.smoker], df.bmi_cat, margins = True)


# In[14]:


fig = px.histogram(df, x='bmi_cat', color='sex', barmode='group', facet_col='smoker')
fig.show()


# In[15]:


a = df.groupby(['sex','smoker','bmi_cat'])['charges'].mean().reset_index()
a.head(24)


# In[16]:


fig = px.bar(a, x="sex", y="charges", color="bmi_cat", barmode='group')
fig.show()


# Here we can observe that the average charges for non-smoking males & females of different bmi categories, is almost 10,000 dollars. Both genders who happen to be smokers as well as any form of obese, have average charges as high as 50,000 dollars. 

# In[17]:


pd.crosstab(df.region, df.bmi_cat, margins = True)


# In[18]:


fig = px.strip(df, x="region", y="charges", color="bmi_cat")
fig.show()


# In[19]:


pd.crosstab(df.region, df.smoker, margins = True)


# In[20]:


fig = px.strip(df, x="region", y="charges", color="smoker")
fig.show()


# We see that the southeaster part of U.S is leading in charges but majority of all customers from all parts of US are charged between 0-20k only.

# In[21]:


fig = px.scatter(df, x="age", y="charges", color="smoker")
fig.show()


# Observations:
# 
# 1. Medical charges increase with age, which is a quite natural trend.
# 
# 2. We can see three clusters of points, each of which seems to form a line with an increasing slope:
# 
# i. The first cluster consists of non-smokers who have relatively low medical charges compared to others.
# 
# ii. The second cluster contains a mix of smokers and non-smokers. It's possible that these are actually two distinct but overlapping clusters: "non-smokers with other medical issues" and "smokers without major medical issues".
# 
# iii. The final cluster consists of smokers with major medical issues that are possibly related to or worsened by smoking.

# In[22]:


fig = px.scatter(df, x='bmi', y='charges', color='smoker')
fig.show()


# It appears that for non-smokers, an increase in BMI doesn't seem to be related to an increase in medical charges. However, medical charges seem to be significantly higher for smokers with a BMI greater than 30.

# In[23]:


fig = px.ecdf(df, x="charges", color="smoker")
fig.show()


# Observation:
# Almost 67% of non-smokers have medical charges of 10,000 dollars or less. Whereas, for smokers, the charges start only from nearly 13,000 dollars, and almost 55% of smokers have medical charges more than 30,000 dollars. 

# In[24]:


fig = px.ecdf(df, x="charges", color="sex")
fig.show()


# It can be seen that almost 61% of both males & females have medical charges of equal to or less than 12,000 dollars. However, after that point, the percentage of females seem to increase by a bit.

# In[25]:


fig = px.scatter_3d(df, x='age', y='bmi', z='charges', color='smoker')
fig.update_traces(marker_size=5, marker_opacity=0.5)
fig.show()


# We can see that it's harder to interpret a 3D scatter plot compared to a 2D scatter plot. As we add more features, it becomes impossible to visualize all feature at once, which is why we use measures like correlation and loss (rmse).

# ## Getting Dummy Variables 

# In[26]:


df.drop(['bmi_cat'],axis=1,inplace=True)


# In[27]:


columns_cat = ['sex','smoker','region']
df_encoded = pd.get_dummies(df, columns=columns_cat)
df_encoded


# ## Correlation Matrix

# In[28]:


plt.figure(figsize=(20,10))
sns.heatmap(df_encoded.corr(), annot=True)
plt.show()


# We see that smoker_yes is highly correlated with charges.
# Also, age and bmi have positive correlation with charges.

# ## Simple Linear Regression

# In[29]:


from sklearn.linear_model import LinearRegression
slr = LinearRegression()


# In[30]:


x = df['age']
y = df['charges']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=100)


# In[31]:


slr.fit(x_train.values.reshape(-1,1), y_train)


# In[32]:


slr.coef_


# In[33]:


slr.intercept_


# In[34]:


slr_pred = slr.predict(x_test.values.reshape(-1,1))


# In[35]:


fig = px.scatter(x=y_test, y=slr_pred, labels={'x':'Actual','y':'Predicted'})
fig.show()


# Majority of the predictions are not close to the actual charges. Hence we need to add more features to create a better model.

# In[36]:


import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test,slr_pred)
rmse = np.sqrt(mean_squared_error(y_test,slr_pred))
r2 = np.abs(r2_score(y_test,slr_pred))

print('MSE:',mse)
print('RMSE:',rmse)
print('R square:',r2)


# Assumptions:
# 1. Linearity Check

# In[37]:


px.scatter(df, x='age', y='charges', trendline='ols', trendline_color_override = 'red')


# 2. Normality of Residuals

# In[38]:


residual = y_test - slr_pred


# In[39]:


import plotly.figure_factory as ff
fig = ff.create_distplot(hist_data=[residual.tolist()], group_labels=['residual'], bin_size=[50])
fig.show()


# 3. Homoscedasticity

# In[40]:


fig = px.scatter(x=slr_pred, y=residual, labels={'x':'Predicted','y':'Residual'})
fig.show()


# 4. No Autocorrelation of Errors

# In[41]:


px.line(list(residual))


# ## Multiple Linear Regression

# In[42]:


X=df_encoded.drop(['charges'], axis=1)
Y=df_encoded['charges']


# In[43]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=100)


# In[44]:


mlr = LinearRegression()
mlr.fit(x_train,y_train)


# In[45]:


mlr_pred = mlr.predict(x_test)


# In[46]:


fig = px.scatter(x=y_test, y=mlr_pred, labels={'x':'Actual','y':'Predicted'})
fig.show()


# In[47]:


mse = mean_squared_error(y_test,mlr_pred)
rmse = np.sqrt(mean_squared_error(y_test,mlr_pred))
r2 = np.abs(r2_score(y_test,mlr_pred))

print('MSE:',mse)
print('RMSE:',rmse)
print('R square:',r2)


# In[48]:


mlr.coef_


# In[49]:


mlr.intercept_


# In[50]:


columns = ['age', 'bmi', 'children', 'smoker_yes', 'smoker_no','sex_male', 'sex_female', 
                 'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']

weights_df_encoded = pd.DataFrame({'features': np.append(columns, 1),'weights': np.append(mlr.coef_ , mlr.intercept_)})
weights_df_encoded


# While it seems like "sex_female" ,"region_northeast","bmi" have a higher weight than age, keep in mind that the range of values for "sex_female" and "region_northeast" columns only take the values 0 and 1, while that of bmi ranges from 15 to 53.
# 
# Because different columns have different ranges, we run into two issues:
# 
# 1. We can't compare the weights of different columns to identify which features are important.
# 2. A column with a larger range of inputs may disproportionately affect the loss and dominate the optimization process.
# For this reason, it's common practice to scale (or standardize) the values in numerical columns.
# 
# We can apply scaling using the StandardScaler class from scikit-learn.

# In[51]:


from sklearn.preprocessing import StandardScaler
numerical_columns = ['age', 'bmi', 'children'] 
scaler = StandardScaler()
scaler.fit(df_encoded[numerical_columns])


# In[52]:


scaled_columns = scaler.transform(df_encoded[numerical_columns])
categorical_columns = ['smoker_yes', 'smoker_no','sex_male', 'sex_female', 
              'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']
categorical_data = df_encoded[categorical_columns].values


# In[53]:


x = np.concatenate((scaled_columns, categorical_data), axis=1)
y = df_encoded.charges


# In[54]:


model = mlr.fit(x, y)
predictions = model.predict(x)


# In[55]:


mse = mean_squared_error( y, predictions )
rmse = np.sqrt(mean_squared_error( y, predictions ))
r2 = np.abs(r2_score( y, predictions ))

print('MSE:',mse)
print('RMSE:',rmse)
print('R square:',r2)


# In[56]:


weights_df_encoded = pd.DataFrame({'features': np.append(numerical_columns + categorical_columns, 1),
                           'weights': np.append(mlr.coef_, mlr.intercept_)})
weights_df_encoded.sort_values('weights', ascending=False)


# As we can see now, the most important features are:
# 1. Smokers
# 2. Age
# 3. BMI

# We need to check the multicollinearity among the significant independent variables, using VIF.

# In[57]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

X = df_encoded[['smoker_yes', 'age', 'bmi', 'region_northeast', 'children', 'sex_female']]

vif_data = pd.DataFrame()
vif_data["features"] = X.columns

vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(len(X.columns))]

print(vif_data)


# As we can see, age and bmi have very high values of VIF, indicating that these two variables are highly correlated. This is expected as the age of a person does influence their bmi. Hence, considering these two features together leads to a model with high multicollinearity.

# In[58]:


X = df_encoded[['smoker_yes', 'age', 'region_northeast', 'children', 'sex_female']]

vif_data = pd.DataFrame()
vif_data["features"] = X.columns

vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(len(X.columns))]

print(vif_data)


# Looking at the VIF values, our model does not exhibit any multicollinearity. Now, we can trust the model coefficients. 

# In[59]:


import statsmodels.api as sm
X = sm.add_constant(df_encoded[['age','smoker_yes','region_northeast','children','sex_female']])
Y = df_encoded['charges']


# In[60]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=100)


# In[61]:


df_encoded_model1 = sm.OLS(y_train,x_train).fit()
df_encoded_model1.summary2()


# Since the P values for region_northeast and sex_female are not-significant, let’s drop these terms to build the final model.

# In[62]:


significant_variables = ['smoker_yes', 'age', 'children']
x_train = x_train[significant_variables] 
x_test = x_test[significant_variables]
df_encoded_model2 = sm.OLS(y_train, x_train).fit() 
df_encoded_model2.summary2() 


# In[63]:


mlr_pred = df_encoded_model2.predict(x_test)


# In[64]:


mse = mean_squared_error(y_test,mlr_pred)
rmse = np.sqrt(mean_squared_error(y_test,mlr_pred))
r2 = np.abs(r2_score(y_test,mlr_pred))

print('MSE:',mse)
print('RMSE:',rmse)
print('R square:',r2)


# Assumptions:
# 1. There must be a linear relationship between the outcome variable and the independent variables.  Scatterplots can show whether there is a linear or curvilinear relationship.

# In[65]:


fig = px.scatter_3d(df_encoded, x='age', y='children', z='charges', color='smoker_yes')
fig.update_traces(marker_size=5, marker_opacity=0.5)
fig.show()


# 2. Multivariate Normality: Multiple regression assumes that the residuals are normally distributed.

# In[66]:


residual = y_test - mlr_pred


# In[67]:


fig = ff.create_distplot(hist_data=[residual.tolist()], group_labels=['residual'], bin_size=[50])
fig.show()


# 3. No Multicollinearity: Multiple regression assumes that the independent variables are not highly correlated with each other.  This assumption is tested using Variance Inflation Factor (VIF) values, which we have already done.

# In[ ]:




