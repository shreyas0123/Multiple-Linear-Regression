################################### problem1 ##########################
import pandas as pd
import numpy as np

#load the dataset
startups_data = pd.read_csv("C:/Users/DELL/Downloads/50_Startups.csv")
startups_data.columns

#Replace the column name
startups_data = startups_data.rename({"R&D Spend" : "RD_Spend","Marketing Spend" : "Marketing_Spend"}, axis  = 'columns')

#EDA Analysis
startups_data.describe()
#label encoding
from sklearn.preprocessing import LabelEncoder
Label_enc = LabelEncoder()
startups_data['State'] = Label_enc.fit_transform(startups_data['State'])
# startups_data['State'] = startups_data['State'].str.replace('New York','0').str.replace('California','1').str.replace('Florida','2')

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# Administration
plt.bar(height = startups_data.Administration, x = np.arange(1, 51, 1))
plt.hist(startups_data.Administration) #histogram
plt.boxplot(startups_data.Administration) #boxplot

# State_California
plt.bar(height = startups_data['RD_Spend'], x = np.arange(1, 51, 1))
plt.hist(startups_data['RD_Spend']) #histogram
plt.boxplot(startups_data['RD_Spend']) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=startups_data['Administration'], y=startups_data['RD_Spend'])

# Countplot
plt.figure(1, figsize=(10, 10))
sns.countplot(startups_data['RD_Spend'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(startups_data.Profit, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(startups_data.iloc[:, :])

# Correlation matrix 
startups_data.corr()

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('Profit ~ RD_Spend + Administration + Marketing_Spend + State', data = startups_data).fit() # regression model

# Summary
ml1.summary()

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)

# Studentized Residuals = Residual/standard deviation of residuals
# index 49,48 is showing high influence so we can exclude that entire row

startups_data_new = startups_data.drop(startups_data.index[[49,48]])

# Preparing model                  
ml_new = smf.ols('Profit ~ RD_Spend + Administration + Marketing_Spend + State', data = startups_data_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables

rsq_RD_Spend = smf.ols('RD_Spend ~ Administration + Marketing_Spend + State', data = startups_data).fit().rsquared  
vif_RD_Spend = 1/(1 - rsq_RD_Spend) 

rsq_Administration = smf.ols('Administration ~ RD_Spend + Marketing_Spend + State', data = startups_data).fit().rsquared  
vif_Administration  = 1/(1 - rsq_Administration)

rsq_Marketing_Spend = smf.ols('Marketing_Spend ~ RD_Spend + Administration + State', data = startups_data).fit().rsquared  
vif_Marketing_Spend = 1/(1 - rsq_Marketing_Spend) 

rsq_State = smf.ols('State ~ Administration + Marketing_Spend + RD_Spend', data = startups_data).fit().rsquared  
vif_State = 1/(1 - rsq_State) 

# Storing vif values in a data frame
d1 = {'Variables':['RD_Spend', 'Administration', 'Marketing_Spend', 'State'], 'VIF':[vif_RD_Spend, vif_Administration, vif_Marketing_Spend,rsq_Marketing_Spend]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
#VIF is not greater than 10 hence there is no colinearity problem arises,no need to remove columns

# Final model
final_ml = smf.ols('Profit ~ RD_Spend + Administration + Marketing_Spend + State', data = startups_data).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(startups_data)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = startups_data.Profit, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
startups_data_train,startups_data_test = train_test_split(startups_data, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("Profit ~ RD_Spend + Administration + Marketing_Spend + State", data = startups_data_train).fit()

# prediction on test data set 
test_pred = model_train.predict(startups_data_test)

# test residual values 
test_resid = test_pred - startups_data_test.Profit
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

# train_data prediction
train_pred = model_train.predict(startups_data_train)

# train residual values 
train_resid  = train_pred - startups_data_train.Profit
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

################################## problem2 #########################################
import pandas as pd
import numpy as np

#load the dataset
computer_data = pd.read_csv("C:/Users/DELL/Downloads/Computer_Data.csv")
computer_data.columns

#EDA Analysis
startups_data.describe()
#drop unnamed column
computer_data.drop(['Unnamed: 0'],axis = 1,inplace  = True)

#label encoding
from sklearn.preprocessing import LabelEncoder
Label_enc = LabelEncoder()
computer_data['cd'] = Label_enc.fit_transform(computer_data['cd'])
computer_data['multi'] = Label_enc.fit_transform(computer_data['multi'])
computer_data['premium'] = Label_enc.fit_transform(computer_data['premium'])
#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# Administration
plt.bar(height = computer_data.speed, x = np.arange(1, 6260, 1))
plt.hist(computer_data.speed) #histogram
plt.boxplot(computer_data.speed) #boxplot

# State_California
plt.bar(height = computer_data['hd'], x = np.arange(1, 6260, 1))
plt.hist(computer_data['hd']) #histogram
plt.boxplot(computer_data['hd']) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=computer_data['speed'], y=computer_data['hd'])

# Countplot
plt.figure(1, figsize=(10, 10))
sns.countplot(computer_data['speed'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(computer_data.price, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(computer_data.iloc[:, :])

# Correlation matrix 
computer_data.corr()

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend' , data = computer_data).fit() # regression model

# Summary
ml1.summary()

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

#sm.graphics.influence_plot(ml1)

# Studentized Residuals = Residual/standard deviation of residuals
# index 49,48 is showing high influence so we can exclude that entire row

#computer_data_new = computer_data.drop(computer_data.index[[49,48]])

# Preparing model                  
ml_new = smf.ols('price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend' , data = computer_data).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables

rsq_speed = smf.ols('speed ~ hd + ram + screen + cd + multi + premium + ads + trend', data = computer_data).fit().rsquared  
vif_speed = 1/(1 - rsq_speed) 

rsq_hd = smf.ols('hd ~ speed + ram + screen + cd + multi + premium + ads + trend', data = computer_data).fit().rsquared  
vif_hd  = 1/(1 - rsq_hd)

rsq_ram = smf.ols('ram ~ speed + hd + screen + cd + multi + premium + ads + trend', data = computer_data).fit().rsquared  
vif_ram = 1/(1 - rsq_ram) 

rsq_screen = smf.ols('screen ~ speed + hd + ram + cd + multi + premium + ads + trend', data = computer_data).fit().rsquared  
vif_screen = 1/(1 - rsq_screen) 

rsq_cd = smf.ols('cd ~ speed + hd + ram + screen + multi + premium + ads + trend', data = computer_data).fit().rsquared  
vif_cd = 1/(1 - rsq_cd) 

rsq_multi = smf.ols('multi ~ speed + hd + ram + cd + cd + premium + ads + trend', data = computer_data).fit().rsquared  
vif_multi = 1/(1 - rsq_multi) 

rsq_premium = smf.ols('premium ~ speed + hd + ram + cd + multi + multi + ads + trend', data = computer_data).fit().rsquared  
vif_premium = 1/(1 - rsq_premium) 

rsq_ads= smf.ols('ads ~ speed + hd + ram + cd + multi + multi + premium + trend', data = computer_data).fit().rsquared  
vif_ads = 1/(1 - rsq_ads) 

rsq_trend = smf.ols('trend ~ speed + hd + ram + cd + multi + multi + ads + ads', data = computer_data).fit().rsquared  
vif_trend = 1/(1 - rsq_trend) 

# Storing vif values in a data frame
d1 = {'Variables':['speed','hd', 'ram', 'screen', 'cd','multi','premium','ads','trend'], 'VIF':[vif_speed, vif_hd, vif_ram,vif_screen,vif_cd,vif_multi,vif_premium,vif_ads,vif_trend]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
#VIF is not greater than 10 hence there is no colinearity problem arises,no need to remove columns

# Final model
final_ml = smf.ols('price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend' , data = computer_data).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(computer_data)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = computer_data.price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
computer_data_train,computer_data_test = train_test_split(computer_data, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend' , data = computer_data).fit()

# prediction on test data set 
test_pred = model_train.predict(computer_data_test)

# test residual values 
test_resid = test_pred - computer_data_test.price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

# train_data prediction
train_pred = model_train.predict(computer_data_train)

# train residual values 
train_resid  = train_pred - computer_data_train.price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

######################################## problem3 ##############################################################
import pandas as pd
import numpy as np

# loading the data
cars = pd.read_csv("C:/Users/DELL/Downloads/ToyotaCorolla.csv" , encoding= 'latin-1')

#using only required features
cols = ['Price','Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight']
cars = cars[cols]
cars.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

#Price
plt.bar(height = cars.Price, x = np.arange(1, 1437, 1))
plt.hist(cars.Price) #histogram
plt.boxplot(cars.Price) #boxplot

# Age_08_04
plt.bar(height = cars.Age_08_04, x = np.arange(1, 1437, 1))
plt.hist(cars.Age_08_04) #histogram
plt.boxplot(cars.Age_08_04) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=cars['Age_08_04'], y=cars['Price'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(cars['KM'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(cars.Price, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(cars.iloc[:, :])
                             
# Correlation matrix 
cars.corr()

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = cars).fit() # regression model

# Summary
ml1.summary()
# p-values for Doors, cc are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 80 is showing high influence so we can exclude that entire row

cars_new = cars.drop(cars.index[[80]])

# Preparing model                  
ml_new = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = cars_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_Age = smf.ols('Age_08_04 ~ KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = cars).fit().rsquared  
vif_Age = 1/(1 - rsq_Age) 

rsq_KM = smf.ols('KM ~ Age_08_04 + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = cars).fit().rsquared  
vif_KM = 1/(1 - rsq_KM)

rsq_HP = smf.ols('HP ~ Age_08_04 + KM + cc + Doors + Gears + Quarterly_Tax + Weight', data = cars).fit().rsquared  
vif_HP = 1/(1 - rsq_HP) 

rsq_cc = smf.ols('cc ~ Age_08_04 + KM + HP + Doors + Gears + Quarterly_Tax + Weight', data = cars).fit().rsquared  
vif_cc = 1/(1 - rsq_cc) 

rsq_Doors = smf.ols('Doors ~ Age_08_04 + KM + HP + cc + Gears + Quarterly_Tax + Weight', data = cars).fit().rsquared  
vif_Doors = 1/(1 - rsq_Doors)

rsq_Gears = smf.ols('Gears ~ Age_08_04 + KM + HP + cc + Doors + Quarterly_Tax + Weight', data = cars).fit().rsquared  
vif_Gears = 1/(1 - rsq_Gears)

rsq_Tax = smf.ols('Quarterly_Tax ~ Age_08_04 + KM + HP + cc + Doors + Gears + Weight', data = cars).fit().rsquared  
vif_Tax = 1/(1 - rsq_Tax)

rsq_Weight = smf.ols('Weight ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax', data = cars).fit().rsquared  
vif_Weight = 1/(1 - rsq_Weight)

# Storing vif values in a data frame
d1 = {'Variables':['Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight'], 'VIF':[vif_Age, vif_KM, vif_HP, vif_cc , vif_Doors , vif_Gears , vif_Tax , vif_Weight]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# Since no VIF value > 10, we are not going to drop any from the prediction model

# Final model
final_ml = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = cars).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(cars)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = cars.Price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
cars_train, cars_test = train_test_split(cars_new, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight", data = cars_train).fit()

# prediction on test data set 
test_pred = model_train.predict(cars_test)

# test residual values 
test_resid = test_pred - cars_test.Price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

# train_data prediction
train_pred = model_train.predict(cars_train)

# train residual values 
train_resid  = train_pred - cars_train.Price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

##########################################Problem 4########################################
import pandas as pd
import numpy as np

# loading the data
avacado_data = pd.read_csv("C:/Users/DELL/Downloads/Avacado_Price.csv")

# Exploratory data analysis:
avacado_data.describe()
#renamimg column
col = {'XLarge Bags': 'XLarge_Bags'}
avacado_data.rename(col , axis = 1 , inplace = True)
#droping unwanted features
avacado_data.drop(['year' , 'region'] , axis = 1 , inplace = True)

#labelEncoding for categprical data
from sklearn import preprocessing
L_enc = preprocessing.LabelEncoder()
avacado_data['type'] = L_enc.fit_transform(avacado_data['type'])

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# AveragePrice
plt.bar(height = avacado_data.AveragePrice, x = np.arange(1, 18250, 1))
plt.hist(avacado_data.AveragePrice) #histogram
plt.boxplot(avacado_data.AveragePrice) #boxplot

# tot_ava1
plt.bar(height = avacado_data.tot_ava1, x = np.arange(1, 18250, 1))
plt.hist(avacado_data.tot_ava1) #histogram
plt.boxplot(avacado_data.tot_ava1) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=avacado_data['tot_ava1'], y=avacado_data['tot_ava3'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(avacado_data['AveragePrice'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(avacado_data.AveragePrice, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(avacado_data.iloc[:,:])
                             
# Correlation matrix 
avacado_data.corr()

# we see there exists High collinearity between input variables especially between
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge_Bags + type', data = avacado_data).fit() # regression model

# Summary
ml1.summary()
# p-values for many columns are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 17468 is showing high influence so we can exclude that entire row

avacado_data_new = avacado_data.drop(avacado_data.index[[17468]])

# Preparing model                  
ml_new = smf.ols('AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge_Bags + type', data = avacado_data_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_Total_Volume = smf.ols('Total_Volume ~ tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge_Bags + type', data = avacado_data).fit().rsquared  
vif_Total_Volume = 1/(1 - rsq_Total_Volume) 

rsq_tot_ava1 = smf.ols('tot_ava1 ~ Total_Volume + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge_Bags + type', data = avacado_data).fit().rsquared  
vif_tot_ava1 = 1/(1 - rsq_tot_ava1)

rsq_tot_ava2 = smf.ols('tot_ava2 ~ Total_Volume + tot_ava1 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge_Bags + type', data = avacado_data).fit().rsquared  
vif_tot_ava2 = 1/(1 - rsq_tot_ava2) 

rsq_tot_ava3 = smf.ols('tot_ava3 ~ Total_Volume + tot_ava1 + tot_ava2 + Total_Bags + Small_Bags + Large_Bags + XLarge_Bags + type', data = avacado_data).fit().rsquared  
vif_tot_ava3 = 1/(1 - rsq_tot_ava3)  

rsq_Total_Bags = smf.ols('Total_Bags ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Small_Bags + Large_Bags + XLarge_Bags + type', data = avacado_data).fit().rsquared  
vif_Total_Bags = 1/(1 - rsq_Total_Bags) 

rsq_Small_Bags = smf.ols('Small_Bags ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Large_Bags + XLarge_Bags + type', data = avacado_data).fit().rsquared  
vif_Small_Bags = 1/(1 - rsq_Small_Bags) 

rsq_Large_Bags = smf.ols('Large_Bags ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + XLarge_Bags + type', data = avacado_data).fit().rsquared  
vif_Large_Bags = 1/(1 - rsq_Large_Bags) 

rsq_XLarge_Bags = smf.ols('XLarge_Bags ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + type', data = avacado_data).fit().rsquared  
vif_XLarge_Bags = 1/(1 - rsq_XLarge_Bags) 

rsq_type = smf.ols('type ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge_Bags', data = avacado_data).fit().rsquared  
vif_type = 1/(1 - rsq_type)

# Storing vif values in a data frame
d1 = {'Variables':['Total_Volume' , 'tot_ava1' , 'tot_ava2' , 'tot_ava3' , 'Total_Bags' , 'Small_Bags' , 'Large_Bags' , 'XLarge_Bags' , 'type'],
      'VIF':[vif_Total_Volume, vif_tot_ava1, vif_tot_ava2, vif_tot_ava3 , vif_Total_Bags, vif_Small_Bags, vif_Large_Bags, vif_XLarge_Bags , vif_type]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As WT is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge_Bags + type', data = avacado_data).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(avacado_data)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = avacado_data.AveragePrice, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)

#Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
train, test = train_test_split(avacado_data, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge_Bags + type", data = train).fit()

# prediction on test data set 
test_pred = model_train.predict(test)

# test residual values 
test_resid = test_pred - test.AveragePrice
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

# train_data prediction
train_pred = model_train.predict(train)

# train residual values 
train_resid  = train_pred - train.AveragePrice
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

#######################################################END########################################

