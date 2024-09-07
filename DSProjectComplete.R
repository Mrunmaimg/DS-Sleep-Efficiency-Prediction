# Load required libraries
library(caret)
library(rpart)
library(randomForest)
library(e1071)
library(gbm)
library(Metrics)

# Read the dataset
f <- read.csv("Sleep_Efficiency_Updated.csv")

# Remove unnecessary columns
f <- f[, -which(names(f) %in% c("Bedtime", "Wakeup.time", "ID"))]

# Check for missing values and impute missing values with mean
f$Awakenings[is.na(f$Awakenings)] <- mean(f$Awakenings, na.rm = TRUE)
f$Caffeine.consumption[is.na(f$Caffeine.consumption)] <- mean(f$Caffeine.consumption, na.rm = TRUE)
f$Alcohol.consumption[is.na(f$Alcohol.consumption)] <- mean(f$Alcohol.consumption, na.rm = TRUE)
f$Exercise.frequency[is.na(f$Exercise.frequency)] <- mean(f$Exercise.frequency, na.rm = TRUE)

# Convert categorical variables to factors
f$Gender <- as.factor(f$Gender)  # Male: 1, Female: 0
f$Smoking.status <- as.factor(f$Smoking.status)  # Yes: 1, No: 0

# Train/test split
set.seed(123)
train_index <- sample(1:nrow(f), 0.7 * nrow(f))
train_data <- f[train_index, ]
test_data <- f[-train_index, ]

#----
ctrl <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
rfe_profile <- rfe(x = train_data[, -which(names(train_data) == "Sleep.efficiency")], 
                   y = train_data$Sleep.efficiency,
                   sizes = c(1:ncol(train_data) - 1),
                   rfeControl = ctrl)
# Get the selected features
selected_features <- predictors(rfe_profile)
# Train a Random Forest model on the selected features
rf_model <- randomForest(Sleep.efficiency ~ ., data = train_data[, c(selected_features, "Sleep.efficiency")])
# Extract variable importance scores
importance_scores <- importance(rf_model)
# Sort importance scores in descending order
sorted_importance <- importance_scores[order(importance_scores, decreasing = TRUE), ]
# Print importance scores
# print(as.list(sorted_importance))
selected_col<- names(sorted_importance)[1:8]
# Random Forest Model
rf_model <- randomForest(Sleep.efficiency ~ ., data = train_data[, c(selected_col, "Sleep.efficiency")])
# Evaluate models
rf_predictions <- predict(rf_model, newdata = test_data[, selected_col])
#------------------------
# Calculate Residual Sum of Squares (RSS) for each model
rf_rss <- sum((rf_predictions - test_data$Sleep.efficiency)^2)
# Calculate Total Sum of Squares (TSS)
mean_y <- mean(test_data$Sleep.efficiency)
tss <- sum((test_data$Sleep.efficiency - mean_y)^2)
# Calculate R-squared for each model
rf_r_squared <- 1 - (rf_rss / tss)
# Print R-squared for each model
cat("Random Forest R-squared:", rf_r_squared, "\n")
