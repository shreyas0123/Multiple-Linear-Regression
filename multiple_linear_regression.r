############################## problem1 ################################################
library(readr)

#load the dataset
startups_data <- read.csv("C:\\Users\\DELL\\Downloads\\50_Startups.csv")
colnames(startups_data)

View(startups_data)

attach(startups_data)

# Normal distribution
qqnorm(R.D.Spend)
qqline(R.D.Spend)

#perform EDA analysis
summary(startups_data)
sum(is.na(startups_data))

#startups_data$State = factor(startups_data$State,levels = c("New York","California","Florida"),labels = c(0,1,2))

#Label Encoding
factors <- factor(startups_data$State)
startups_data$State <- as.numeric(factors )

str(startups_data)

# Scatter plot
plot(Administration, Profit) # Plot relation ships between each X with Y
plot(Marketing.Spend, Profit)

# Or make a combined plot
pairs(startups_data)   # Scatter plot for all pairs of variables
plot(startups_data)

cor(Administration, Profit)
cor(startups_data) # correlation matrix

# The Linear Model of interest
model.startups_data <- lm(Profit ~ R.D.Spend + Administration + Marketing.Spend + State, data = startups_data) # lm(Y ~ X)
summary(model.startups_data)

model.startupsRD <- lm(Profit ~ R.D.Spend)
summary(model.startupsRD)

model.startupsADM <- lm(Profit ~ Administration)
summary(model.startupsADM)

model.startupsRD_ADM <- lm(Profit ~ R.D.Spend + Administration)
summary(model.startupsRD_ADM)

#### Scatter plot matrix with Correlations inserted in graph
install.packages("GGally")
library(GGally)
ggpairs(startups_data)

### Partial Correlation matrix
install.packages("corpcor")
library(corpcor)
cor(startups_data)

cor2pcor(cor(startups_data))

# Diagnostic Plots
install.packages("car")
library(car)

plot(model.startups_data)# Residual Plots, QQ-Plot, Std. Residuals vs Fitted, Cook's distance

qqPlot(model.startups_data, id.n = 5) # QQ plots of studentized residuals, helps identify outliers

# Deletion Diagnostics for identifying influential obseravations
influenceIndexPlot(model.startups_data, id.n = 3) # Index Plots of the influence measures
influencePlot(model.startups_data, id.n = 3) # A user friendly representation of the above

# Regression after deleting the 49th and 46th observation
model.startups_data1 <- lm(Profit ~ R.D.Spend + Administration + Marketing.Spend + State, data = startups_data[-49,-46 ])
summary(model.startups_data1)

### Variance Inflation Factors
vif(model.startups_data)  # VIF is > 10 => collinearity

# Regression model to check R^2 on Independent variales
VIFRD <- lm(R.D.Spend ~ Administration + Marketing.Spend + State,data = startups_data)
VIFAD <- lm(Administration ~ R.D.Spend + Marketing.Spend + State,data = startups_data)
VIFMARK <- lm(Marketing.Spend ~ R.D.Spend + Administration + State,data = startups_data)
VIFS <- lm(State ~ R.D.Spend + Administration + Marketing.Spend,data = startups_data)

summary(VIFRD)
summary(VIFAD)
summary(VIFMARK)
summary(VIFS)

# VIF of Marketing.Spend
1/(1-0.5422)

#### Added Variable Plots ######
avPlots(model.startups_data, id.n = 2, id.cex = 0.8, col = "red")

# Linear Model without State, state in avplots depends on output
model.final <- lm(Profit ~ R.D.Spend + Administration + Marketing.Spend, data = startups_data)
summary(model.final)

# Linear model without WT and influential observation
model.final1 <- lm(Profit ~ R.D.Spend + Administration + Marketing.Spend, data = startups_data[-46 ])
summary(model.final1)

# Added Variable Plots
avPlots(model.final1, id.n = 2, id.cex = 0.8, col = "red")

# Variance Influence Plot
vif(model.final1)

# Evaluation Model Assumptions
plot(model.final1)
plot(model.final1$fitted.values, model.final1$residuals)

qqnorm(model.final1$residuals)
qqline(model.final1$residuals)

# Subset selection
# 1. Best Subset Selection
# 2. Forward Stepwise Selection
# 3. Backward Stepwise Selection / Backward Elimination

install.packages("leaps")
library(leaps)
lm_best <- regsubsets(Profit ~ ., data = startups_data, nvmax = 15)
summary(lm_best)

summary(lm_best)$adjr2
which.max(summary(lm_best)$adjr2)
coef(lm_best, 2)

lm_forward <- regsubsets(Profit ~ ., data = startups_data, nvmax = 15, method = "forward")
summary(lm_forward)

# Data Partitioning
n <- nrow(startups_data)
n1 <- n * 0.7
n2 <- n - n1
train <- sample(1:n, n1)
test <- startups_data[-train, ]

# Model Training
model.final <- lm(Profit ~ R.D.Spend + Administration + Marketing.Spend , data = startups_data[train,])
summary(model.final)

pred <- predict(model.final, newdata = test)
actual <- test$Profit
error <- actual - pred

test.rmse <- sqrt(mean(error**2))
test.rmse

strain.rmse <- sqrt(mean(model$residuals**2))
train.rmse

# Step AIC
install.packages("MASS")
library(MASS)
stepAIC(model.startups_data)


################################### problem2 ####################################################
library(readr)

#load the dataset
computer_data <- read.csv("C:\\Users\\DELL\\Downloads\\Computer_Data.csv")
colnames(computer_data)

#dropping 1st column
computer_data <- computer_data[-1]

View(computer_data)

attach(computer_data)

# Normal distribution
qqnorm(ram)
qqline(ram)

#perform EDA analysis
summary(computer_data)
sum(is.na(computer_data))

#Label Encoding
factors <- factor(computer_data$cd)
computer_data$cd <- as.numeric(factors )

factors <- factor(computer_data$multi)
computer_data$multi <- as.numeric(factors )

factors <- factor(computer_data$premium)
computer_data$premium <- as.numeric(factors )

str(computer_data)

# Scatter plot
plot(speed, price) # Plot relation ships between each X with Y
plot(ram, price)

# Or make a combined plot
pairs(computer_data)   # Scatter plot for all pairs of variables
plot(computer_data)

cor(speed, price)
cor(computer_data) # correlation matrix

# The Linear Model of interest
model.computer_data <- lm(price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend, data = computer_data) # lm(Y ~ X)
summary(model.computer_data)

model.speed <- lm(price ~ speed)
summary(model.speed)

model.hd <- lm(price ~ hd)
summary(model.hd)

model.speed_hd <- lm(price ~ speed + hd)
summary(model.speed_hd)

#### Scatter plot matrix with Correlations inserted in graph
install.packages("GGally")
library(GGally)
ggpairs(startups_data)

### Partial Correlation matrix
install.packages("corpcor")
library(corpcor)
cor(computer_data)

cor2pcor(cor(computer_data))

# Diagnostic Plots
install.packages("car")
library(car)

plot(model.computer_data)# Residual Plots, QQ-Plot, Std. Residuals vs Fitted, Cook's distance

qqPlot(model.computer_data, id.n = 5) # QQ plots of studentized residuals, helps identify outliers

# Deletion Diagnostics for identifying influential obseravations
influenceIndexPlot(model.computer_data, id.n = 3) # Index Plots of the influence measures
influencePlot(model.computer_data, id.n = 3) # A user friendly representation of the above

### Variance Inflation Factors
vif(model.computer_data)  # VIF is > 10 => collinearity

# Regression model to check R^2 on Independent variales
VIFSP <- lm(speed ~ hd + ram + screen + cd + multi + premium + ads + trend, data = computer_data)
VIFHD <- lm(hd ~ speed + ram + screen + cd + multi + premium + ads + trend, data = computer_data)
VIFRM <- lm(ram ~ speed + hd + screen + cd + multi + premium + ads + trend, data = computer_data)
VIFSC <- lm(screen ~ speed + ram + hd + cd + multi + premium + ads + trend, data = computer_data)
VIFCD <- lm(cd ~ speed + ram + screen + hd + multi + premium + ads + trend, data = computer_data)
VIFML <- lm(multi ~ speed + ram + screen + cd + hd + premium + ads + trend, data = computer_data)
VIFPR <- lm(premium ~ speed + ram + screen + cd + multi + hd + ads + trend, data = computer_data)
VIFAD <- lm(ads ~ speed + ram + screen + cd + multi + premium + hd + trend, data = computer_data)
VIFTR <- lm(trend ~ speed + ram + screen + cd + multi + premium + ads + hd, data = computer_data)

summary(VIFSP)
summary(VIFHD)
summary(VIFRM)
summary(VIFSC)
summary(VIFCD)
summary(VIFML)
summary(VIFPR)
summary(VIFAD)
summary(VIFTR)

# VIF of trend
1/(1-0.505)

#### Added Variable Plots ######
avPlots(model.computer_data, id.n = 2, id.cex = 0.8, col = "red")

# Linear Model without CD , CD depends on output in avplots
model.final <- lm(price ~ speed + hd + ram + screen + multi + premium + ads + trend, data = computer_data)
summary(model.final)

# Linear model without CD and influential observation
model.final1 <- lm(price ~ speed + hd + ram + screen + multi + premium + ads + trend, data = computer_data)
summary(model.final1)

# Added Variable Plots
avPlots(model.final1, id.n = 2, id.cex = 0.8, col = "red")

# Variance Influence Plot
vif(model.final1)

# Evaluation Model Assumptions
plot(model.final1)
plot(model.final1$fitted.values, model.final1$residuals)

qqnorm(model.final1$residuals)
qqline(model.final1$residuals)

# Subset selection
# 1. Best Subset Selection
# 2. Forward Stepwise Selection
# 3. Backward Stepwise Selection / Backward Elimination

install.packages("leaps")
library(leaps)
lm_best <- regsubsets(price ~ ., data = computer_data, nvmax = 15)
summary(lm_best)

summary(lm_best)$adjr2
which.max(summary(lm_best)$adjr2)
coef(lm_best, 9)

lm_forward <- regsubsets(price ~ ., data = computer_data, nvmax = 15, method = "forward")
summary(lm_forward)

# Data Partitioning
n <- nrow(computer_data)
n1 <- n * 0.7
n2 <- n - n1
samp <- sample(1:n, n1)
train <- computer_data[samp, ]
test <- computer_data[-samp, ]

# Model Training
model.final <- lm(price ~ speed + hd + ram + screen + multi + premium + ads + trend, data = train)
summary(model.final)

pred <- predict(model.final, newdata = test)
actual <- test$price
error <- actual - pred

test.rmse <- sqrt(mean(error**2))
test.rmse

train.rmse <- sqrt(mean(model.final$residuals**2))
train.rmse

# Step AIC
install.packages("MASS")
library(MASS)
stepAIC(model.startups_data)

############################### problem3 ####################################################
# Load the Cars dataset
library(readr)
car_data <- read.csv("C:\\Users\\DELL\\Downloads\\ToyotaCorolla.csv", header = TRUE )
View(car_data)

attach(car_data)

# Normal distribution
qqnorm(Price)
qqline(Price)

# Exploratory data analysis:
summary(car_data)

#feature selection
cols <- c('Price','Age_08_04', 'KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight')
car_data <- car_data[cols]

# Scatter plot
plot(Price, KM) # Plot relation ships between each X with Y
plot(cc, Weight)

# Or make a combined plot
pairs(car_data)   # Scatter plot for all pairs of variables
plot(car_data)

cor(cc, Weight)
cor(car_data) # correlation matrix

# The Linear Model of interest
model.car <- lm(Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight, data = car_data) # lm(Y ~ X)
summary(model.car)

model.carage <- lm(Price ~ Age_08_04)
summary(model.carage)

model.carKM <- lm(Price ~ KM)
summary(model.carKM)

model.carHP <- lm(Price ~ HP)
summary(model.carHP)

model.carcc <- lm(Price ~ cc)
summary(model.carcc)

model.carDr <- lm(Price ~ Doors)
summary(model.carDr)

model.carGr <- lm(Price ~ Gears)
summary(model.carGr)

model.carQ <- lm(Price ~ Quarterly_Tax)
summary(model.carQ)

model.carW <- lm(Price ~ Weight)
summary(model.carW)

#Scatter plot matrix with Correlations inserted in graph
install.packages("GGally")
library(GGally)
ggpairs(car_data)

### Partial Correlation matrix
install.packages("corpcor")
library(corpcor)
cor(car_data)

cor2pcor(cor(car_data))

# Diagnostic Plots
install.packages("car")
library(car)

plot(model.car)# Residual Plots, QQ-Plot, Std. Residuals vs Fitted, Cook's distance

qqPlot(model.car, id.n = 5) # QQ plots of Standardized residuals, helps identify outliers

# Deletion Diagnostics for identifying influential observations
influenceIndexPlot(model.car, id.n = 3) # Index Plots of the influence measures
influencePlot(model.car, id.n = 3) # A user friendly representation of the above

# Regression after deleting the 81th observation
model.car1 <- lm(Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight, data = car_data[-81, ])
summary(model.car1)

### Variance Inflation Factors
vif(model.car)  # VIF is > 10 => col-linearity

#Added Variable Plots for model.comp
avPlots(model.car, id.n = 2, id.cex = 0.8, col = "red")

# Added Variable Plots for model.comp1
avPlots(model.car1, id.n = 2, id.cex = 0.8, col = "red")

# Variance Influence Plot
vif(model.car1)

#model after removing Doors
final.model <- lm(Price ~ Age_08_04 + KM + HP + cc + Gears + Quarterly_Tax + Weight, data = car_data[-81, ])
summary(final.model)

# Evaluation Model Assumptions
plot(final.model)
plot(final.model$fitted.values, final.model$residuals)

qqnorm(final.model$residuals)
qqline(final.model$residuals)

# Subset selection
install.packages("leaps")
library(leaps)

#Best Subset Selection
lm_best <- regsubsets(Price ~ ., data = car_data, nvmax = 15)
summary(lm_best)
summary(lm_best)$adjr2
which.max(summary(lm_best)$adjr2)
coef(lm_best, 3)

#forward subset selection
lm_forward <- regsubsets(Price ~ ., data = car_data, nvmax = 15, method = "forward")
summary(lm_forward)

# Data Partitioning
n <- nrow(car_data)
n1 <- n * 0.7
n2 <- n - n1
samp <- sample(1:n, n1)
train <- car_data[samp , ]
test <- car_data[-samp, ]

# Model Training without Doors
model <- lm(Price ~ Age_08_04 + KM + HP + cc + Gears + Quarterly_Tax + Weight, data = train)
summary(model)

#predicting on test data
pred <- predict(model, newdata = test)
actual <- test$Price
error <- actual - pred

#rmse of error on test data
test.rmse <- sqrt(mean(error**2))
test.rmse

#train rmse
train.rmse <- sqrt(mean(model$residuals**2))
train.rmse

# Step AIC
install.packages("MASS")
library(MASS)
stepAIC(model)

####################################################problem 4#####################################
# Load the Cars dataset
library(readr)
avacado_data <- read.csv("C:\\Users\\DELL\\Downloads\\Avacado_Price.csv", header = TRUE )
View(avacado_data)

attach(avacado_data)

# Normal distribution
qqnorm(AveragePrice)
qqline(AveragePrice)

# Exploratory data analysis:
summary(avacado_data)

#label encoding
fact_type <-  as.factor(type)
avacado_data$type <- as.numeric(fact_type)


#dropping unwanted column
avacado_data <- avacado_data[,1:10]

# Scatter plot
plot(tot_ava1, tot_ava2) # Plot relation ships between each X with Y
plot(tot_ava3, tot_ava1)

# Or make a combined plot
pairs(avacado_data)   # Scatter plot for all pairs of variables
plot(avacado_data)

cor(AveragePrice, XLarge.Bags)
cor(avacado_data) # correlation matrix

# The Linear Model of interest
model.avacado <- lm(AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge.Bags + type, data = avacado_data) # lm(Y ~ X)
summary(model.avacado)

model.avacadoTV <- lm(AveragePrice ~ Total_Volume)
summary(model.avacadoTV)

model.avacadotv1 <- lm(AveragePrice ~ tot_ava1)
summary(model.avacadotv1)

model.avacadotv2 <- lm(AveragePrice ~ tot_ava2)
summary(model.avacadotv2)

model.avacadotv3 <- lm(AveragePrice ~ tot_ava3)
summary(model.avacadotv3)

model.avacadoTB <- lm(AveragePrice ~ Total_Bags)
summary(model.avacadoTB)

model.avacadoSB <- lm(AveragePrice ~ Small_Bags)
summary(model.avacadoSB)

model.avacadoLB <- lm(AveragePrice ~ Large_Bags)
summary(model.avacadoLB)

model.avacadoXB <- lm(AveragePrice ~ XLarge.Bags)
summary(model.avacadoXB)

model.avacadot <- lm(AveragePrice ~ type)
summary(model.avacadot)

#Scatter plot matrix with Correlations inserted in graph
install.packages("GGally")
library(GGally)
ggpairs(avacado_data)

### Partial Correlation matrix
install.packages("corpcor")
library(corpcor)
cor(avacado_data)

cor2pcor(cor(comp_data))

# Diagnostic Plots
install.packages("car")
library(car)

plot(model.avacado)# Residual Plots, QQ-Plot, Std. Residuals vs Fitted, Cook's distance

qqPlot(model.avacado, id.n = 5) # QQ plots of Standardized residuals, helps identify outliers

# Deletion Diagnostics for identifying influential observations
influenceIndexPlot(model.avacado, id.n = 3) # Index Plots of the influence measures
influencePlot(model.avacado, id.n = 3) # A user friendly representation of the above

# Regression after deleting the 15561,17469 , 5486 ,14126,17429 observation
model.avacado1 <- lm(AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge.Bags + type, data = avacado_data[-c(15561,17469 , 5486 ,14126,17429), ])
summary(model.avacado1)

### Variance Inflation Factors
vif(model.avacado)  # VIF is > 10 => col-linearity

#Added Variable Plots for model.comp
avPlots(model.avacado, id.n = 2, id.cex = 0.8, col = "red")
#Total_Bags , Small_Bags ,Large_Bags, XLarge.Bags are depended on output variable so can be removed

# Added Variable Plots for model.comp1
avPlots(model.avacado1, id.n = 2, id.cex = 0.8, col = "red")

# Variance Influence Plot
vif(model.avacado1)

# Evaluation Model Assumptions
plot(model.avacado)
plot(model.avacado$fitted.values, model.avacado$residuals)

qqnorm(model.avacado$residuals)
qqline(model.avacado$residuals)

# Subset selection
install.packages("leaps")
library(leaps)

#Best Subset Selection
lm_best <- regsubsets(AveragePrice ~ ., data = avacado_data, nvmax = 15)
summary(lm_best)
summary(lm_best)$adjr2
which.max(summary(lm_best)$adjr2)
coef(lm_best, 3)

#forward subset selection
lm_forward <- regsubsets(AveragePrice ~ ., data = avacado_data, nvmax = 15, method = "forward")
summary(lm_forward)

# Data Partitioning
n <- nrow(avacado_data)
n1 <- n * 0.7
n2 <- n - n1
samp <- sample(1:n, n1)
train <- avacado_data[samp , ]
test <- avacado_data[-samp, ]

# Model Training
model <- lm(AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3  + type, data = train)
summary(model)

#predicting on test data
pred <- predict(model, newdata = test)
actual <- test$AveragePrice
error <- actual - pred

#rmse of error on test data
test.rmse <- sqrt(mean(error**2))
test.rmse

#train rmse
train.rmse <- sqrt(mean(model$residuals**2))
train.rmse

# Step AIC
install.packages("MASS")
library(MASS)
stepAIC(model)
############################################################END###################################





