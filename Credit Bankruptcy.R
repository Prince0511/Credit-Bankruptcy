#Libraries
library(aod)
library(caret)
library(caTools)
library(dplyr)
library(ggplot2)
library(ISLR)
library(mlbench)
library(plyr)
library(readr)
library(stats)
library(MASS)
library(rpart)
library(rpart.plot)
--------------------------------------------------------------------------------
#Loading the dataset
library(readxl)
credit <- read_excel("C:/Users/kheni/OneDrive/Desktop/credit.xlsx")
View(credit)
  
#Checking dimensions and summary of data
dim(credit)                                     # 1000 * 17
summary(credit)
names(credit)

#Factoring 'Default' variable
credit$default <- as.factor(credit$default)
credit$default

#Combining the columns which makes sense
columns <- c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17)
credit_updated <- credit[,columns]
credit_updated
dim(credit_updated)                               # 1000 * 17
--------------------------------------------------------------------------------

#Data Splitting (Training and Testing)
set.seed(1000)
row.number = sample(1:nrow(credit_updated), 0.8*nrow(credit_updated))
train = credit_updated[row.number,]
test = credit_updated[-row.number,]
dim(train)                                        # 800 * 9
dim(test)                                         # 200 * 9
--------------------------------------------------------------------------------
  
--------------------------------------------------------------------------------
# LOGISTIC REGRESSION
--------------------------------------------------------------------------------
ctrl <- trainControl(method = "cv", number = 10, savePredictions = TRUE) 

model_glm = train(default ~ .,             
                  data = train,                               
                  method = "glm",
                  family = "binomial",
                  metric = "Accuracy",                      
                  trControl = ctrl,                     
                  preProcess = c("center","scale"),
                  tuneLength = 20)

model_glm

glm_predict <- predict(model_glm ,newdata = test)

confusionMatrix(glm_predict, test$default, mode = 'prec_recall', positive = 'yes')
--------------------------------------------------------------------------------

ctrl <- trainControl(method = "cv", number = 10, savePredictions = TRUE) 

model_glm_1 = train(default ~ checking_balance + months_loan_duration + credit_history                               + savings_balance  + amount   + employment_duration                                    +  percent_of_income + other_credit + years_at_residence                               + amount/months_loan_duration + age +                                                  + housing + purpose + existing_loans_count,                                           data = train,                               
                    method = "glm",
                    family = "binomial",
                    metric = "Accuracy",                      
                    trControl = ctrl,                     
                    preProcess = c("center","scale"),
                    tuneLength = 20)

model_glm_1

glm_predict_1 <- predict(model_glm_1 ,newdata = test)

confusionMatrix(glm_predict_1, test$default, mode = 'prec_recall', positive = 'yes')
--------------------------------------------------------------------------------

ctrl <- trainControl(method = "cv", number = 10, savePredictions = TRUE) 

model_glm_2 = train(default ~. -amount -employment_duration -job,
                    data = train,                               
                    method = "glm",
                    family = "binomial",
                    metric = "Accuracy",                      
                    trControl = ctrl,                     
                    preProcess = c("center","scale"),
                    tuneLength = 20)

model_glm_2

glm_predict_2 <- predict(model_glm_2 ,newdata = test)

confusionMatrix(glm_predict_2, test$default, mode = 'prec_recall', positive = 'yes')


--------------------------------------------------------------------------------
# LINEAR DISCRIMINANT ANALYSIS (LDA)
--------------------------------------------------------------------------------
ctrl <- trainControl(method = "cv", number = 10, savePredictions = TRUE) 

model_lda = train(default ~ .,             
                  data = train,                               
                  method = "lda",
                  family = "binomial",
                  metric = "Accuracy",                      
                  trControl = ctrl,                     
                  preProcess = c("center","scale"),
                  tuneLength = 20)

model_lda

lda_predict <- predict(model_lda ,newdata = test)

confusionMatrix(lda_predict, test$default, mode = 'prec_recall', positive = 'yes')

--------------------------------------------------------------------------------
ctrl <- trainControl(method = "cv", number = 10, savePredictions = TRUE) 

model_lda_1 = train(default ~ checking_balance + months_loan_duration + credit_history                               + savings_balance  + amount   + employment_duration                                    +  percent_of_income + other_credit + years_at_residence                               + amount/months_loan_duration + age +                                                  + housing + purpose + existing_loans_count,                                           data = train,                               
                  method = "lda",
                  family = "binomial",
                  metric = "Accuracy",                      
                  trControl = ctrl,                     
                  preProcess = c("center","scale"),
                  tuneLength = 20)

model_lda_1

lda_predict_1 <- predict(model_lda_1 ,newdata = test)

confusionMatrix(lda_predict_1, test$default, mode = 'prec_recall', positive = 'yes')

--------------------------------------------------------------------------------
ctrl <- trainControl(method = "cv", number = 10, savePredictions = TRUE) 

model_lda_2 = train(default ~ checking_balance + months_loan_duration + credit_history                               + savings_balance  + amount   + employment_duration                                    +  percent_of_income + other_credit +                                                  + amount/months_loan_duration + age +                                                  + housing + purpose + existing_loans_count,             
                    data = train,                               
                    method = "lda",
                    family = "binomial",
                    metric = "Accuracy",                      
                    trControl = ctrl,                     
                    preProcess = c("center","scale"),
                    tuneLength = 20)

model_lda_2

lda_predict_2 <- predict(model_lda_2 ,newdata = test)

confusionMatrix(lda_predict_2, test$default, mode = 'prec_recall', positive = 'yes')
--------------------------------------------------------------------------------

ctrl <- trainControl(method = "cv", number = 10, savePredictions = TRUE) 

model_lda_3 = train(default ~ . -amount -employment_duration -job +amount/months_loan_duration -phone -age -years_at_residence,             
                    data = train,                               
                    method = "lda",
                    family = "binomial",
                    metric = "Accuracy",                      
                    trControl = ctrl,                     
                    preProcess = c("center","scale"),
                    tuneLength = 20)

model_lda_3

lda_predict_3 <- predict(model_lda_3 ,newdata = test)

confusionMatrix(lda_predict_3, test$default, mode = 'prec_recall', positive = 'yes')

--------------------------------------------------------------------------------
# Quadratic Discriminant Analysis (QDA)
--------------------------------------------------------------------------------
ctrl <- trainControl(method = "cv", number = 10, savePredictions = TRUE) 

model_qda_1 = train(default ~.,             
                    data = train,                               
                    method = "qda",
                    family = "binomial",
                    metric = "Accuracy",                      
                    trControl = ctrl,                     
                    preProcess = c("center","scale"),
                    tuneLength = 20)

model_qda_1

qda_predict_1 <- predict(model_qda_1 ,newdata = test)

confusionMatrix(qda_predict_1, test$default, mode = 'prec_recall', positive = 'yes')

--------------------------------------------------------------------------------
  
ctrl <- trainControl(method = "cv", number = 10, savePredictions = TRUE) 

model_qda_2 = train(default ~ checking_balance + months_loan_duration + credit_history                               + savings_balance  + amount   + employment_duration                                    +  percent_of_income + other_credit                                                    + amount/months_loan_duration + age +                                                  + housing + purpose + existing_loans_count,             
                    data = train,                               
                    method = "qda",
                    family = "binomial",
                    metric = "Accuracy",                      
                    trControl = ctrl,                     
                    preProcess = c("center","scale"),
                    tuneLength = 20)

model_qda_2

qda_predict_2 <- predict(model_qda_2 ,newdata = test)

confusionMatrix(qda_predict_2, test$default, mode = 'prec_recall', positive = 'yes')
--------------------------------------------------------------------------------

ctrl <- trainControl(method = "cv", number = 10, savePredictions = TRUE) 

model_qda_3 = train(default ~. -amount -job -credit_history -percent_of_income -years_at_residence,             
                    data = train,                               
                    method = "qda",
                    family = "binomial",
                    metric = "Accuracy",                      
                    trControl = ctrl,                     
                    preProcess = c("center","scale"),
                    tuneLength = 20)

model_qda_3

qda_predict_3 <- predict(model_qda_3 ,newdata = test)

confusionMatrix(qda_predict_3, test$default, mode = 'prec_recall', positive = 'yes')



--------------------------------------------------------------------------------
# K Nearest Neighbor (KNN)
--------------------------------------------------------------------------------
ctrl <- trainControl(method = "cv", number = 10, savePredictions = TRUE) 
  
model_knn_1 <- train(default ~ checking_balance + months_loan_duration + credit_history                               + savings_balance  + amount   + employment_duration                                    +  percent_of_income + other_credit                                                    + amount/months_loan_duration + age +                                                  + housing + purpose + existing_loans_count,             
                  data = train,                              
                  method = "knn",                             
                  #family="binomial",                   
                  metric = "Accuracy",                     
                  trControl = ctrl,                     
                  preProcess = c("center","scale"),   
                  tuneLength = 20)

model_knn_1

plot(model_knn_1)

knn_predict_1 <- predict(model_knn_1, newdata = test)

confusionMatrix(knn_predict_1, test$default, mode = 'prec_recall', positive = 'yes')
--------------------------------------------------------------------------------

ctrl <- trainControl(method = "cv", number = 10, savePredictions = TRUE) 

model_knn_2 <- train(default ~. -amount -job -years_at_residence +amount/months_loan_duration -purpose,             
                     data = train,                              
                     method = "knn",                             
                     #family="binomial",                   
                     metric = "Accuracy",                     
                     trControl = ctrl,                     
                     preProcess = c("center","scale"),   
                     tuneLength = 20)

model_knn_2

knn_predict_2 <- predict(model_knn_2, newdata = test)

confusionMatrix(knn_predict_2, test$default, mode = 'prec_recall', positive = 'yes')



--------------------------------------------------------------------------------
# Decision Tree
--------------------------------------------------------------------------------
ctrl <- trainControl(method = "cv", number = 10, savePredictions = TRUE) 

model_tree_1 <- train(default ~ .,             
                     data = train,                              
                     method = "rpart",                             
                     #family="binomial",                   
                     metric = "Accuracy",                     
                     trControl = ctrl,                     
                     preProcess = c("center","scale"),   
                     tuneLength = 20)

model_tree_1

tree_predict_1 <- predict(model_tree_1, newdata = test)

confusionMatrix(tree_predict_1, test$default, mode = 'prec_recall', positive = 'yes')

--------------------------------------------------------------------------------
ctrl <- trainControl(method = "cv", number = 10, savePredictions = TRUE) 

model_tree_2 <- train(default ~ checking_balance + months_loan_duration +                                             credit_history + savings_balance  + amount   +                                        employment_duration  +  percent_of_income + other_credit                               + amount/months_loan_duration + age +  + housing +                                     purpose + existing_loans_count,             
                      data = train,                              
                      method = "rpart",                             
                      #family="binomial",                   
                      metric = "Accuracy",                     
                      trControl = ctrl,                     
                      preProcess = c("center","scale"),   
                      tuneLength = 20)

model_tree_2

tree_predict_2 <- predict(model_tree_2, newdata = test)

confusionMatrix(tree_predict_2, test$default, mode = 'prec_recall', positive = 'yes')

--------------------------------------------------------------------------------
ctrl <- trainControl(method = "cv", number = 10, savePredictions = TRUE) 

model_tree_2 <- train(default ~. -amount -job -years_at_residence -percent_of_income,                        data = train,                              
                      method = "rpart",                             
                      #family="binomial",                   
                      metric = "Accuracy",                     
                      trControl = ctrl,                     
                      preProcess = c("center","scale"),   
                      tuneLength = 20)

model_tree_2

tree_predict_2 <- predict(model_tree_2, newdata = test)

confusionMatrix(tree_predict_2, test$default, mode = 'prec_recall', positive = 'yes')


tree<- rpart(default ~. -amount -job -years_at_residence -percent_of_income,                        data = train, method = 'class')

rpart.plot(tree, extra = 106)
