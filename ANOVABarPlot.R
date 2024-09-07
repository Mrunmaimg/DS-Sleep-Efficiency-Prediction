library(rpart)
library(randomForest)
library(e1071)
library(gbm)
library(caret)
library(glmnet)
library(Metrics)

# Read the dataset
f <- read.csv("Sleep_Efficiency_Updated.csv")

# Remove unnecessary columns
f <- f[, -which(names(f) %in% c("Bedtime", "Wakeup.time", "ID"))]

# Check for missing values
AllCols <- colSums(is.na(f))
total_na <- sum(is.na(f))

# Impute missing values with mean
f$Awakenings[is.na(f$Awakenings)] <- mean(f$Awakenings, na.rm = TRUE)
f$Caffeine.consumption[is.na(f$Caffeine.consumption)] <- mean(f$Caffeine.consumption, na.rm = TRUE)
f$Alcohol.consumption[is.na(f$Alcohol.consumption)] <- mean(f$Alcohol.consumption, na.rm = TRUE)
f$Exercise.frequency[is.na(f$Exercise.frequency)] <- mean(f$Exercise.frequency, na.rm = TRUE)

# Check for missing values after imputation
AllCols <- colSums(is.na(f))

# Convert categorical variables to factors
f$Gender <- as.factor(f$Gender)  # Male: 1, Female: 0
f$Smoking.status <- as.factor(f$Smoking.status)  # Yes: 1, No: 0

# Train/test split
set.seed(123)
train_index <- sample(1:nrow(f), 0.7 * nrow(f))
train_data <- f[train_index, ]
test_data <- f[-train_index, ]

cat("Column Names and Numbers:\n")
print(data.frame(Name = names(f), Number = 1:ncol(f)), row.names = FALSE)
#------
# Perform ANOVA to select features
anova_result <- anova(lm(Sleep.efficiency ~ ., data = train_data))

# Select features with p-value less than 0.05
selected_features <- rownames(anova_result)[anova_result$`Pr(>F)` < 0.05]

# Ensure selected features are present in training data
selected_features <- intersect(selected_features, colnames(train_data))

# Sort selected features based on their p-values (importance) in descending order
sorted_features <- selected_features[order(anova_result[selected_features, "Pr(>F)"], decreasing = TRUE)]

# Print sorted features and their p-values (importance scores)
cat("Sorted Features based on Importance (p-values) in Descending Order:\n")
cat(sorted_features, sep = "\n")

cat("\nImportance Scores (p-values) in Descending Order:\n")
cat(anova_result[sorted_features, "Pr(>F)"], sep = "\n")


#------
# Perform ANOVA to select features
anova_result <- anova(lm(Sleep.efficiency ~ ., data = train_data))

# Select features with p-value less than 0.05
selected_features <- rownames(anova_result)[anova_result$`Pr(>F)` < 0.05]

# Ensure selected features are present in training data
selected_features <- intersect(selected_features, colnames(train_data))
sorted_features <- rownames(anova_result)[order(anova_result$`Pr(>F)`)]

# Select top 5 features
top_features <- sorted_features[1:8]
selected_features<-top_features
#-----

# Decision Tree Model with selected features
dt_model <- rpart(Sleep.efficiency ~ ., data = train_data[, c("Sleep.efficiency", selected_features)], method = "anova")

# Random Forest Model with selected features
rf_model <- randomForest(Sleep.efficiency ~ ., data = train_data[, c("Sleep.efficiency", selected_features)])

# Support Vector Regression (SVR) Model with selected features
svm_model <- svm(Sleep.efficiency ~ ., data = train_data[, c("Sleep.efficiency", selected_features)])

# Gradient Boosting Model with selected features
gb_model <- gbm(Sleep.efficiency ~ ., data = train_data[, c("Sleep.efficiency", selected_features)], distribution = "gaussian")

# Evaluate models (e.g., using RMSE for regression)
dt_predictions <- predict(dt_model, newdata = test_data)
rf_predictions <- predict(rf_model, newdata = test_data)
svm_predictions <- predict(svm_model, newdata = test_data)
gb_predictions <- predict(gb_model, newdata = test_data)

# Print RMSE 
cat("Decision Tree RMSE:", dt_rmse, "\n")
cat("Random Forest RMSE:", rf_rmse, "\n")
cat("SVR RMSE:", svm_rmse, "\n")
cat("Gradient Boosting RMSE:", gb_rmse, "\n")

# Evaluate MAE for each model
dt_mae <- mae(dt_predictions, test_data$Sleep.efficiency)
rf_mae <- mae(rf_predictions, test_data$Sleep.efficiency)
svm_mae <- mae(svm_predictions, test_data$Sleep.efficiency)
gb_mae <- mae(gb_predictions, test_data$Sleep.efficiency)

# Print MAE 
cat("\nDecision Tree MAE:", dt_mae, "\n")
cat("Random Forest MAE:", rf_mae, "\n")
cat("SVR MAE:", svm_mae, "\n")
cat("Gradient Boosting MAE:", gb_mae, "\n")

# Calculate R-squared for each model
dt_rss <- sum((dt_predictions - test_data$Sleep.efficiency)^2)
rf_rss <- sum((rf_predictions - test_data$Sleep.efficiency)^2)
svm_rss <- sum((svm_predictions - test_data$Sleep.efficiency)^2)
gb_rss <- sum((gb_predictions - test_data$Sleep.efficiency)^2)

mean_y <- mean(test_data$Sleep.efficiency)
tss <- sum((test_data$Sleep.efficiency - mean_y)^2)

dt_r_squared <- 1 - (dt_rss / tss)
rf_r_squared <- 1 - (rf_rss / tss)
svm_r_squared <- 1 - (svm_rss / tss)
gb_r_squared <- 1 - (gb_rss / tss)

# Print R-squared for each model
cat("\nDecision Tree R-squared:", dt_r_squared, "\n")
cat("Random Forest R-squared:", rf_r_squared, "\n")
cat("SVR R-squared:", svm_r_squared, "\n")
cat("Gradient Boosting R-squared:", gb_r_squared, "\n")

#for (feature in top_features) {
# importance_score <- anova_result[feature, "Pr(>F)"]
# cat("Feature:", feature, "Importance Score:", importance_score, "\n")
# Load required libraries
library(ggplot2)

# Create a data frame with model names and their corresponding metrics
metrics_df <- data.frame(
  Model = c("Decision Tree", "Random Forest", "SVR", "Gradient Boosting"),
  RMSE = c(dt_rmse, rf_rmse, svm_rmse, gb_rmse),
  MAE = c(dt_mae, rf_mae, svm_mae, gb_mae),
  R_squared = c(dt_r_squared, rf_r_squared, svm_r_squared, gb_r_squared)
)

# Melt the data frame to long format for plotting
melted_metrics <- reshape2::melt(metrics_df, id.vars = "Model", variable.name = "Metric", value.name = "Value")

# Create a bar plot
barplot <- ggplot(melted_metrics, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = round(Value, 2)), vjust = -0.5, position = position_dodge(width = 0.9)) +  # Add data labels
  labs(title = "Performance Metrics of ANOVA",
       y = "Value", fill = "Metric") +
  theme_minimal()

# Display the bar plot
print(barplot)
