if (!require(tidyverse)) install.packages('tidyverse')
library(tidyverse)
if (!require(caret)) install.packages('caret')
library(caret)
if (!require(factoextra)) install.packages('factoextra')
library(factoextra)
if (!require(randomForest)) install.packages('randomForest')
library(randomForest)
if (!require(data.table)) install.packages('data.table')
library(data.table)
if (!require(knitr)) install.packages('knitr')
library(knitr)
if (!require(rpart)) install.packages('rpart')
library(rpart)


# Load the data
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

# Sets names to the values from the website (https://archive.ics.uci.edu/dataset/2/adult)
column_names <- c("age", "workclass", "fnlwgt", "education", "education_num", "marital_status", 
                  "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", 
                  "hours_per_week", "native_country", "income")

# Load the data & replaces NA with "?"
data <- read.csv(url, header = FALSE, col.names = column_names, na.strings = "?")
sum(is.na(data))

##############################################################
# Data exploration 
##############################################################

### Data Exploration ###

## Above/Under 50K count
data %>% 
  group_by(income) %>%
  summarize(count = n())

# Age exploration
data %>% 
  ggplot(aes(x=as.factor(age), fill=income)) +
  geom_bar(position = "stack") +
  labs(x = "Age", y = "Count", title = "Count of Income Levels by Age") +
  scale_x_discrete(breaks = seq(0, 110, by = 5))

# Workclass exploration
data %>% 
  ggplot(aes(x=workclass, fill=income)) +
  geom_bar(position = "dodge") +
  labs(x = "Workclass", y = "Percent breakdown", title = "Workclass exploration") 

# fnlwgt exploration
data %>% 
  ggplot(aes(x = fnlwgt, fill = factor(income))) +
  geom_density(alpha = 0.5) +
  scale_x_log10() +
  labs(x = "Final Weight", y = "Density", title = "Final weight exploration")

# education exploration
data %>% 
  ggplot(aes(x=education, fill=income)) +
  geom_bar(position = "dodge") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Education", y = "Count", title = "Education exploration")

# education-num exploration
data %>% 
  ggplot(aes(x=as.factor(`education_num`), fill=income)) +
  geom_bar(position = "dodge") +
  labs(x = "Education_Num", y = "Count", title = "Education_Num exploration")

# marital-status exploration
data %>%
  ggplot(aes(x=as.factor(`marital_status`), fill=income)) +
  geom_bar(position = "dodge") +
  coord_flip() +
  labs(x = "Marital Status", y = "Count", title = "Marital Status exploration")

# occupation exploration
data %>%
  ggplot(aes(x=as.factor(occupation), fill=income)) +
  geom_bar(position = "dodge") +
  coord_flip() +
  labs(x = "Occupation", y = "Count", title = "Occupation exploration")

# relationship exploration
data %>%
  ggplot(aes(x=as.factor(relationship), fill=income)) +
  geom_bar(position = "dodge") +
  coord_flip() +
  labs(x = "Relationship", y = "Count", title = "Relationship exploration")

# race exploration
data %>%
  ggplot(aes(x=as.factor(race), fill=income)) +
  geom_bar(position = "dodge") +
  coord_flip() +
  labs(x = "Race", y = "Count", title = "Race exploration")

# sex (M is 1 F is 0) exploration
data <- data %>%
  mutate(sex = as.factor(sex),
         sex = ifelse(sex == " Male",TRUE,FALSE)) %>%
  rename(sex_M = sex)

data %>%
  ggplot(aes(x=sex_M, fill=income)) +
  geom_bar(position = "dodge") +
  labs(x = "Male Sex (True is Male)", y = "Count", title = "Sex exploration")

# capital-gain exploration
data %>%
  ggplot(aes(x = `capital_gain`, fill = factor(income))) +
  geom_density(alpha = 0.5) +
  scale_x_log10() +
  labs(x = "Capital Gain", y = "Count", title = "Capital Gain exploration")

# capital-loss exploration
data %>%
  ggplot(aes(x=`capital_loss`, fill = factor(income))) +
  geom_density(alpha = 0.5) +
  scale_x_log10() +
  labs(x = "Capital Loss", y = "Count", title = "Capital Loss exploration")

# hours-per-week exploration
data %>%
  ggplot(aes(x=as.factor(`hours_per_week`), fill=income)) +
  geom_bar(position = "stack") +
  scale_y_log10() +
  scale_x_discrete(breaks = seq(0, 100, by = 10)) +
  labs(x = "Hours per Week Worked", y = "Count", title = "Hours per Week Worked exploration")

#native-country exploration
data %>%
  ggplot(aes(x=`native_country`, fill=income)) +
  geom_bar(position = "dodge") +
  scale_y_log10() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Native Country", y = "Count", title = "Native Country exploration") 

# Changes data to factor for analysis 
data <- data %>%
  mutate(age = as.factor(age),
         workclass = as.factor(workclass),
         education = as.factor(education),
         education_num = as.factor(education_num),
         `marital_status`= as.factor(`marital_status`),
         occupation = as.factor(occupation),
         relationship = as.factor(relationship),
         race = as.factor(race),
         hours_per_week = as.factor(hours_per_week),
         `native_country` = as.factor(`native_country`))

# Removes Final Weight variable
data <- data %>%
  select(-fnlwgt)

# Makes income a factor
data <- data %>%
  mutate(income = as.factor(income))

# Saves original income names
orig_income_names <- unique(data$income)

# Edits factor level names for income to make them more machine learning friendly
levels(data$income) <- make.names(levels(data$income))


##############################################################
# Splits data into training and test sets
##############################################################
# Creates data partition
set.seed(1)
test_index <- createDataPartition(data$income, times = 1, p = 0.2, list = FALSE)

train_df <- data[-test_index,]
test_df_1 <- data[test_index,]

### Ensure that the test data only contains data present in the train ###

test_df <- test_df_1 %>%
  semi_join(train_df, by = c("age", "workclass", "education", "occupation", "native_country"))

### Adds removed test rows to training data ###

removed_data <- anti_join(test_df_1,test_df)
train_df <- rbind(train_df,removed_data)

rm(removed_data, test_df_1, test_index, url, column_names)


##############################################################
# Evaluates data based off of the training set using the
# Random Forest model
##############################################################
# Define the training control
set.seed(5)
train_control <- trainControl(method = "cv", number = 3, classProbs = TRUE, 
                              summaryFunction = twoClassSummary)  # 3-fold cross-validation & sets binary classifier model 

# Creates rfm machine learning model
rf_fit <- train(income ~ ., data = train_df, method = "rf", 
                trControl = train_control, tuneLength = 5, metric = "ROC")

predict_rfm <- predict(rf_fit, newdata = test_df)

# Checks accuracy, precision (PPV), recall (sensitivity/TPR), prevalence, and balanced accuracy (avg of specificity and sensitivity)
RF_confusionMatrix <- confusionMatrix(data = predict_rfm, reference = test_df$income)
RF_confusionMatrix

#F_1 score b=1 default
RF_F_1 <- F_meas(data = predict_rfm, reference = test_df$income)
RF_F_1


# Tabulates model results
results_table <- tibble(Machine_Learning_Model = "Random Forest",
                        Accuracy = RF_confusionMatrix$overall["Accuracy"],
                        `Balanced Accuracy`= RF_confusionMatrix$byClass["Balanced Accuracy"],
                        `F_1 Score` = RF_F_1, 
                        Sensitivity = RF_confusionMatrix$byClass["Sensitivity"], 
                        Specificity = RF_confusionMatrix$byClass["Specificity"],
                        PPV = RF_confusionMatrix$byClass["Pos Pred Value"], 
                        `<=50K Prevalence`= RF_confusionMatrix$byClass["Prevalence"])

results_table


##############################################################
# Evaluates data based off of the training set using the
# XGBoost model
##############################################################
# Set up training control for xgb
set.seed(5)
train_control <- trainControl(method = "cv", number = 3, classProbs = TRUE, 
                              summaryFunction = twoClassSummary) # 3 fold cross validation

#Creates model & predictions
xgb_fit <- train(income ~ ., data = train_df, method = "xgbTree", 
                 trControl = train_control, metric = "ROC")

predict_xgb <- predict(xgb_fit, newdata = test_df)

# Checks accuracy, precision (PPV), recall (sensitivity/TPR), prevalence, and balanced accuracy (avg of specificity and sensitivity)
XGB_confusionMatrix <- confusionMatrix(data = predict_xgb, reference = test_df$income)

XGB_confusionMatrix


#F_1 score b=1 default
XGB_F_1 <- F_meas(data = predict_xgb, reference = test_df$income)

# Tabulates model results
results_table <- rbind(results_table,tibble(Machine_Learning_Model = "XGBoost model",
                                            `Balanced Accuracy`= XGB_confusionMatrix$byClass["Balanced Accuracy"],
                                            Accuracy = XGB_confusionMatrix$overall["Accuracy"],
                                            `F_1 Score` = XGB_F_1, 
                                            Sensitivity = XGB_confusionMatrix$byClass["Sensitivity"], 
                                            Specificity = XGB_confusionMatrix$byClass["Specificity"],
                                            PPV = XGB_confusionMatrix$byClass["Pos Pred Value"], 
                                            `<=50K Prevalence`= XGB_confusionMatrix$byClass["Prevalence"]))

results_table


##############################################################
# Evaluates data based off of the training set using the
# KNN Model
##############################################################
# Define the training control using 3-fold cross-validation
set.seed(5)

train_control <- trainControl(method = "cv", number = 3, classProbs = TRUE, 
                              summaryFunction = twoClassSummary)

tune <- data.frame(k = seq(1, 20, by = 2))

# Makes KNN model 
knn_fit <- train(income ~ ., data = train_df, method = "knn", tuneGrid = tune, 
                 trControl = train_control, metric = "ROC")
knn_fit$bestTune

# Predicts with KNN model
predict_knn <- predict(knn_fit, newdata = test_df)

# Checks accuracy, precision (PPV), recall (sensitivity/TPR), prevalence, and balanced accuracy (avg of specificity and sensitivity)
KNN_confusionMatrix <- confusionMatrix(data = predict_knn, reference = test_df$income)

KNN_confusionMatrix

#F_1 score b=1 default
KNN_F_1 <- F_meas(data = predict_knn, reference = test_df$income)

# Tabulates model results
results_table <- rbind(results_table,tibble(Machine_Learning_Model = "KNN",
                                            Accuracy = KNN_confusionMatrix$overall["Accuracy"],
                                            `Balanced Accuracy`= KNN_confusionMatrix$byClass["Balanced Accuracy"],
                                            `F_1 Score` = KNN_F_1, 
                                            Sensitivity = KNN_confusionMatrix$byClass["Sensitivity"], 
                                            Specificity = KNN_confusionMatrix$byClass["Specificity"],
                                            PPV = KNN_confusionMatrix$byClass["Pos Pred Value"], 
                                            `<=50K Prevalence`= KNN_confusionMatrix$byClass["Prevalence"]))

results_table


##############################################################
# Evaluates data based off of the training set using the
# Decision Tree Model
##############################################################
# Define the training control & tuning parameters using 3-fold cross-validation
set.seed(5)

train_control <- trainControl(method = "cv", number = 3, classProbs = TRUE, 
                              summaryFunction = twoClassSummary)

tune <- expand.grid(cp = seq(0, 0.05, by = 0.01))

# Makes Decision Tree model 
rpart_fit <- train(income ~ ., data = train_df, method = "rpart", tuneGrid = tune, 
                 trControl = train_control, metric = "ROC")

# Predicts with KNN model
predict_rpart <- predict(rpart_fit, newdata = test_df)

# Checks accuracy, precision (PPV), recall (sensitivity/TPR), prevalence, and balanced accuracy (avg of specificity and sensitivity)
RPART_confusionMatrix <- confusionMatrix(data = predict_rpart, reference = test_df$income)

RPART_confusionMatrix

#F_1 score b=1 default
RPART_F_1 <- F_meas(data = predict_rpart, reference = test_df$income)

# Tabulates model results
results_table <- rbind(results_table,tibble(Machine_Learning_Model = "Decision Tree",
                                            Accuracy = RPART_confusionMatrix$overall["Accuracy"],
                                            `Balanced Accuracy`= RPART_confusionMatrix$byClass["Balanced Accuracy"],
                                            `F_1 Score` = RPART_F_1, 
                                            Sensitivity = RPART_confusionMatrix$byClass["Sensitivity"], 
                                            Specificity = RPART_confusionMatrix$byClass["Specificity"],
                                            PPV = RPART_confusionMatrix$byClass["Pos Pred Value"], 
                                            `<=50K Prevalence`= RPART_confusionMatrix$byClass["Prevalence"]))

results_table


##############################################################
# Creates an Ensemble of the models.
# Then evaluates the data based off of the training set
##############################################################

# Calculates ensemble
set.seed(5)

ensemble <- cbind(rfm = predict_rfm == "X...50K", XGBoost = predict_xgb == "X...50K", KNN = predict_knn == "X...50K",rpart = predict_rpart == "X...50K") #1 = "X...50K" (<=50K)

# Predicts "X...50K" (<=50K), X..50K (>50K)
ensemble_predict <- ifelse(rowMeans(ensemble) >= .5, "X...50K", "X..50K")

# Checks accuracy, precision (PPV), recall (sensitivity/TPR), prevalence, and balanced accuracy (avg of specificity and sensitivity)
ensemble_confusionMatrix <- confusionMatrix(data = as.factor(ensemble_predict), reference = test_df$income)

ensemble_confusionMatrix

#F_1 score b=1 default
ENSEMBLE_F_1 <- F_meas(data = as.factor(ensemble_predict), reference = test_df$income)

# Tabulates model results
results_table <- rbind(results_table,tibble(Machine_Learning_Model = "RFM, XGB, KNN, & Rpart Ensemble",
                                            Accuracy = ensemble_confusionMatrix$overall["Accuracy"],
                                            `Balanced Accuracy`= ensemble_confusionMatrix$byClass["Balanced Accuracy"],
                                            `F_1 Score` = ENSEMBLE_F_1, 
                                            Sensitivity = ensemble_confusionMatrix$byClass["Sensitivity"], 
                                            Specificity = ensemble_confusionMatrix$byClass["Specificity"],
                                            PPV = ensemble_confusionMatrix$byClass["Pos Pred Value"], 
                                            `<=50K Prevalence`= ensemble_confusionMatrix$byClass["Prevalence"]))

results_table 

