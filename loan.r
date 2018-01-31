# Dataset and description from 
# https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/

rm(list=ls(all=TRUE)) #workspace cleanup
library(ggplot2); library(ggthemes)   #for visualization
library(mice)                         #for imputation
library(dplyr)                        #for data handling
library(glmx)                         #for running GLM model (logistic regression)
library(randomForest)                 #for random forest
library(evtree)                       #for classification trees
library(e1071)                        #for Naive bayesian model
library(caret)                        #for validating and finetuning models

#install.packages("")                #for installing any missing libraries

#setting working directory
setwd("D:/Study/OneDrive - The University of Texas at Dallas/Winter break 1/Loan application")

# Rough notes
# train + test > full > impute > full_imp > train2, test2
# train2 > testing + training1

#importing data
train <- read.csv('train.csv',stringsAsFactors = T,na.strings=c(""))
test <- read.csv('test.csv',stringsAsFactors = T,na.strings=c(""))
var_desc <- read.csv('vars.csv',stringsAsFactors = F,na.strings=c(""))

#checking if data structure is okay. Changing some variables to factors
str(train)
# train$Loan_Amount_Term <- as.factor(train$Loan_Amount_Term)
# test$Loan_Amount_Term <- as.factor(test$Loan_Amount_Term)
train$Credit_History <- as.factor(train$Credit_History)
test$Credit_History <- as.factor(test$Credit_History)
factor(train$Loan_Status)
levels(train$Loan_Status)

#variable descriptions
var_desc

#summary of training and testing data
head(train)
summary(train)

head(test)
summary(test)

#missing data
cbind(sapply(train,function(x) sum(is.na(x))))
cbind(sapply(test,function(x) sum(is.na(x))))

#There is a lot of missing data in the dataset. I will try to address
#this issue later on.

#Using visualizations to explore the dataset

#people with 2 dependents have a higher loan acceptance rate.
ggplot(train, aes(x = Dependents, fill = factor(Loan_Status))) +
  geom_bar(stat='count', position='fill') +
  labs(x = 'Dependents', y = 'proportion')

#Gender doesnt seem to effect loan status independently
ggplot(train, aes(x = Gender, fill = factor(Loan_Status))) +
  geom_bar(stat='count', position='fill') +
  labs(x = 'Gender', y = 'proportion')

#Married people have a higher success rate
ggplot(train, aes(x = Married, fill = factor(Loan_Status))) +
  geom_bar(stat='count', position='fill') +
  labs(x = 'Married', y = 'proportion')

#No credit history is almost certainly a loan rejection
ggplot(train, aes(x = as.factor(Credit_History), fill = factor(Loan_Status))) +
  geom_bar(stat='count', position='fill') +
  labs(x = 'Credit History', y = 'proportion')

#Graduates also seem to have a higher success rate
ggplot(train, aes(x = Education, fill = factor(Loan_Status))) +
  geom_bar(stat='count', position='fill') +
  labs(x = 'Education', y = 'proportion')

#Self employed doesnt seem to matter either
ggplot(train, aes(x = Self_Employed, fill = factor(Loan_Status))) +
  geom_bar(stat='count', position='fill') +
  labs(x = 'Self_Employed', y = 'proportion')

#credit history can be a very good estimator of loan status
ggplot(train, aes(x = as.factor(Credit_History), fill = factor(Loan_Status))) +
  geom_bar(stat='count', position='fill') +
  labs(x = 'Credit_History', y = 'proportion')

#semi urban property area also has a better success rate
ggplot(train, aes(x = as.factor(Property_Area), fill = factor(Loan_Status))) +
  geom_bar(stat='count', position='fill') +
  labs(x = 'Property_Area', y = 'proportion')

##Applicant income is similar
ggplot(train, aes(x = Loan_Status, y = ApplicantIncome, fill = Loan_Status)) +
                 geom_boxplot() + ylim(0,7000) + 
                  geom_hline(aes(yintercept = 3520 ))

## red > Loan_Status = N
ggplot() + 
  geom_density( data = subset(train,train$Loan_Status=="N"), 
               color = 'darkred', aes(x=ApplicantIncome), fill ='darkred', alpha = 0.4) + 
  geom_density( data = subset(train,train$Loan_Status=="Y"), 
                color = 'darkblue', aes(x=ApplicantIncome)) + xlim(0,10000)

#coapplicant income in failed loan applications seem to be clustered
#around 0, the next chart will explore this further
ggplot(train, aes(x = Loan_Status, y = CoapplicantIncome, fill = Loan_Status)) +
  geom_boxplot() + geom_hline(aes(yintercept=0)) + ylim(0,7000)

##The distribution is not very different
## red > Loan_Status = N
ggplot() + 
  geom_density( data = subset(train,train$Loan_Status=="N"), 
                color = 'darkred', aes(x=CoapplicantIncome), fill ='darkred', alpha = 0.4) + 
  geom_density( data = subset(train,train$Loan_Status=="Y"), 
                color = 'darkblue', aes(x=CoapplicantIncome)) + 
  xlim(0,10000)

#Loan amount also seems to not affect loan status.
ggplot(train, aes(x = Loan_Status, y = LoanAmount, fill = Loan_Status)) +
  geom_boxplot() + geom_hline(aes(yintercept=0))

ggplot(train, aes(x = Loan_Status, y = LoanAmount, fill = Loan_Status)) +
  geom_boxplot() + geom_hline(aes(yintercept=0)) + facet_grid(.~Credit_History)

#applicant and coapplicant incomes dont provide a good separation between Loan Status
ggplot(train, aes(x= train$ApplicantIncome, y=train$CoapplicantIncome, color=train$Loan_Status)) +
  geom_point(position = 'jitter', size = 2) + xlim(0,23000) + ylim(0,12000)

## Coming back to the missing values. 
#Combining the two datasets will possibly give a better estimate of missing values

str(train)
str(test)

full <- bind_rows(train, test)
str(full)
full$Married <- as.factor(full$Married)

cbind(sapply(full,function(x) sum(is.na(x))))

ggplot(train, aes(x = Married, y = CoapplicantIncome, fill = Married)) +
  geom_boxplot() + geom_hline(aes(yintercept=0))

# Married people are generally more likely to have a co applicant income so 
# we will use this information to impute values for Married

#Red > Married
ggplot() + 
  geom_density( data = subset(full,full$Married=="Yes"), 
                color = 'darkred', aes(x=CoapplicantIncome), fill ='darkred', alpha = 0.4) + 
  geom_density( data = subset(full,full$Married=="No"), 
                color = 'darkblue', aes(x=CoapplicantIncome)) + 
  xlim(0,10000)

subset(full, is.na(full$Married))
#one of these 3 people have a coapplicant income, assuming this person to be married
# and the other 3 to be not married
full$Married[105] <- "Yes"
full$Married[229] <- "No"
full$Married[436] <- "No"

#using random forests to impute missing values
set.seed(176)
mice <- mice(full[, !names(full) %in% c('Loan_Id','Loan_Status')], method='rf') 
full_imp <- complete(mice)

#no more missing values
cbind(sapply(full_imp,function(x) sum(is.na(x))))



#Getting back our train and test datasets
library(data.table)
train2 <- full_imp[1:614,]
train2$Loan_Status <- train$Loan_Status

#fixing the data table from imputation
str(train2)
write.csv(train2, file = 'temp.csv', row.names = F)
train2 <- read.csv('temp.csv')

test2 <- full_imp[615:981,]
write.csv(test2, file = 'temp.csv', row.names = F)
test2 <- read.csv('temp.csv')

#dividing the train dataset into training and testing sets.
#the naming is a bit confusing. Because we dont have the labels
#for the test set provided, we can't tune our model. So we have
#to divide the train set given into training and testing datasets.
set.seed(176)
indx <- createDataPartition(train$ApplicantIncome, p = .7, list = F)
training <- train2[indx, ]
#cbind(sapply(training,function(x) sum(is.na(x))))
testing  <- train2[-indx, ]

str(train)
str(training)

#running an untrained model with default parameters
set.seed(176)
model01 <- randomForest(Loan_Status~Gender+Married+Dependents+Education+Self_Employed+ApplicantIncome+CoapplicantIncome+LoanAmount+Loan_Amount_Term+Credit_History+Property_Area, 
                        data = training, ntree = 50) 
model01

#Untrained model has a 22% misclassification rate. which not good enough
#lets try to improve this.
(acc1 <- 1 - mean(predict(model01) == training$Loan_Status))

#using caret to hypertune randomforest
set.seed(176)
model03 <- train(Loan_Status~Gender+Married+Dependents+Education+Self_Employed+ApplicantIncome+CoapplicantIncome+LoanAmount+Loan_Amount_Term+Credit_History+Property_Area, 
                    data=training,
                    ntree = 9, # number of trees, changing this also helps improve accuracy at times 
                    method = "rf") # rf = random forests
model03
#Misclassification error has gone down to 11.1%. This is much better
# but such a low number may imply overfiiting
(acc1 <- 1 - mean(predict(model03) == training$Loan_Status))

#Now checking on the testing set
#Base model: 19.2%
(tacc1 <- 1 - mean(predict(model01, testing) == testing$Loan_Status))
#Tuned model: 20.8%, which means that the base model was better. hmmm
#This possibly means that our model is overfitting to the data
(tacc3 <- 1 - mean(predict(model03, newdata= testing) == testing$Loan_Status))

#changing the number of trees
#to 50, error rate drops down to 19.8%, still worse than base model
set.seed(176)
model04 <- train(Loan_Status~Gender+Married+Dependents+Education+Self_Employed+ApplicantIncome+CoapplicantIncome+LoanAmount+Loan_Amount_Term+Credit_History+Property_Area, 
                 data=training,
                 ntree = 50, # number of trees, changing can affect accuracy 
                 method = "rf") # rf = random forests
model04

(tacc4 <- 1 - mean(predict(model04, newdata= testing) == testing$Loan_Status))

#changing ntree to 5
#still at 20.3%
set.seed(192)
model05 <- train(Loan_Status~Gender+Married+Dependents+Education+Self_Employed+ApplicantIncome+CoapplicantIncome+LoanAmount+Loan_Amount_Term+Credit_History+Property_Area, 
                 data=training,
                 ntree = 5, # number of trees, changing this also helps improve accuracy at times 
                 method = "rf") # rf = random forests
model05
(tacc5 <- 1 - mean(predict(model05, newdata= testing) == testing$Loan_Status))

#lets try a logisitc regression model
set.seed(176)
model06 <- glm(Loan_Status~Gender+Married+Dependents+Education+Self_Employed+ApplicantIncome+CoapplicantIncome+LoanAmount+Loan_Amount_Term+Credit_History+Property_Area, 
               data=training, family=binomial(link="logit"))
summary(model06)
(tacc6 <- 1 - mean(round(predict(model06, newdata= testing, type="response")) == as.numeric(testing$Loan_Status)-1))

set.seed(176)
model07 <- glm(Loan_Status~Married+Credit_History+Property_Area, 
               data=training, family=binomial(link="logit"))
(tacc7 <- 1 - mean(round(predict(model07, newdata= testing,type="response")) == as.numeric(testing$Loan_Status)-1))

model08 <- train(Loan_Status~Gender+Married+Dependents+Education+Self_Employed+ApplicantIncome+CoapplicantIncome+LoanAmount+Loan_Amount_Term+Credit_History+Property_Area, 
                data=training, method = "glm", family = binomial)
model08
(tacc8 <- 1 - mean((predict(model08, newdata= testing,type="raw")) == (testing$Loan_Status)))

submissionmodel <- train(Loan_Status~Gender+Married+Dependents+Education+Self_Employed+ApplicantIncome+CoapplicantIncome+LoanAmount+Loan_Amount_Term+Credit_History+Property_Area, 
                         data=train2,
                         ntree = 50, # number of trees, changing this also helps improve accuracy at times 
                         method = "rf") # rf = random forests
submissionmodel
(acc <- 1 - mean(predict(submissionmodel) == train2$Loan_Status))

#Trying k fold validation, need to read up more about this
set.seed(1567)
ctrl <- trainControl(method = "repeatedcv", number = 5, savePredictions = TRUE)
model09 <- train(Loan_Status~Gender+Married+Dependents+Education+Self_Employed+ApplicantIncome+CoapplicantIncome+LoanAmount+Loan_Amount_Term+Credit_History+Property_Area, 
                 data=train2,
                 ntree = 8, 
                 method = "rf",
                 trControl = ctrl, tuneLength = 5)
model09
pred <- predict(model09, newdata=train2)
confusionMatrix(data=pred, train2$Loan_Status)

Loan_Status <- predict(model09, test2)
solution <- data.frame(Loan_ID = test2$Loan_ID, Loan_Status)
write.csv(solution, file = 'submission.csv', row.names = F)

#Ideas for the future:

#Feature engineering ideas:
#total income = applicant + coapplicant income
#value of each payment
#value of loan respective to total income
#value of monthly payment respective to total income
# co applicant earns or not, binary var

#taking log of incomes as they have a lot of outliers

#kfold validation
#Code from: 
# https://machinelearningmastery.com/how-to-estimate-model-accuracy-in-r-using-the-caret-package/
# # load the library
# library(caret)
# # load the iris dataset
# data(iris)
# # define training control
# train_control <- trainControl(method="cv", number=10)
# # fix the parameters of the algorithm
# grid <- expand.grid(.fL=c(0), .usekernel=c(FALSE))
# # train the model
# model <- train(Species~., data=iris, trControl=train_control, method="nb", tuneGrid=grid)
# # summarize results
# print(model)