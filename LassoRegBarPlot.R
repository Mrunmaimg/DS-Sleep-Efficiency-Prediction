
library(caret)
library(rpart)
library(randomForest)
library(e1071)
library(gbm)
library(Metrics)
library(glmnet)

f <- read.csv("Sleep_Efficiency_Updated.csv")
f <- f[, -which(names(f) %in% c("Bedtime", "Wakeup.time", "ID"))]
f$Awakenings[is.na(f$Awakenings)] <- mean(f$Awakenings, na.rm = TRUE)
f$Caffeine.consumption[is.na(f$Caffeine.consumption)] <- mean(f$Caffeine.consumption, na.rm = TRUE)
f$Alcohol.consumption[is.na(f$Alcohol.consumption)] <- mean(f$Alcohol.consumption, na.rm = TRUE)
f$Exercise.frequency[is.na(f$Exercise.frequency)] <- mean(f$Exercise.frequency, na.rm = TRUE)
f$Gender <- as.factor(f$Gender)  # Male: 1, Female: 0
f$Smoking.status <- as.factor(f$Smoking.status)  # Yes: 1, No: 0

# Train/test split
set.seed(123)
train_index <- sample(1:nrow(f), 0.7 * nrow(f))
train_data <- f[train_index, ]
test_data <- f[-train_index, ]
#----
# Feature selection with Lasso Regression
x <- model.matrix(Sleep.efficiency ~ ., data = train_data)[,-1] # Remove the intercept column
y <- train_data$Sleep.efficiency
lasso_model <- cv.glmnet(x, y, alpha = 1)  # Lasso regression with alpha = 1 for L1 regularization

# Extract selected features
lasso_coef <- coef(lasso_model, s = "lambda.min")  # Extract coefficients for optimal lambda
coef_df <- data.frame(Coefficient = as.numeric(lasso_coef), Feature = c("(Intercept)", names(lasso_coef)[-1]))
coef_df <- coef_df[order(-abs(coef_df$Coefficient)), ]
lasso_selected_indices <- which(lasso_coef != 0)  # Indices of selected features
top_features <- coef_df[abs(coef_df$Coefficient) != 0, ]
top_features <- top_features[order(-abs(top_features$Coefficient)), ]
top_features<-coef_df
cat("Top 5 Features after Lasso Regression:\n")
print(head(top_features, 11))
# Display column names and their corresponding numbers
cat("Column Names and Numbers:\n")
print(data.frame(Name = names(f), Number = 1:ncol(f)), row.names = FALSE)


dt_model <- rpart(Sleep.efficiency ~ ., data = train_data[, c("Age","Light.sleep.percentage","Awakenings","Alcohol.consumption","Smoking.status","Deep.sleep.percentage", "Sleep.efficiency")], method = "anova")
#----
# Random Forest Model
rf_model <- randomForest(Sleep.efficiency ~ ., data = train_data[, c("Age","Light.sleep.percentage","Awakenings","Alcohol.consumption","Smoking.status","Gender","Deep.sleep.percentage", "Sleep.efficiency")])

# Support Vector Regression (SVR) Model
svm_model <- svm(Sleep.efficiency ~ ., data = train_data[, c("Age","Light.sleep.percentage","Awakenings","Alcohol.consumption","Smoking.status","Deep.sleep.percentage","Sleep.efficiency")])

# Gradient Boosting Model
gb_model <- gbm(Sleep.efficiency ~ ., data = train_data[, c("Age","Light.sleep.percentage","Awakenings","Alcohol.consumption","Smoking.status","Deep.sleep.percentage", "Sleep.efficiency")], distribution = "gaussian")
#----
# Evaluate models
dt_predictions <- predict(dt_model, newdata = test_data[, c("Age","Light.sleep.percentage","Awakenings","Alcohol.consumption","Smoking.status","Deep.sleep.percentage","Gender")])
rf_predictions <- predict(rf_model, newdata = test_data[, c("Age","Light.sleep.percentage","Awakenings","Alcohol.consumption","Smoking.status","Gender","Deep.sleep.percentage")])
svm_predictions <- predict(svm_model, newdata = test_data[, c("Age","Light.sleep.percentage","Awakenings","Alcohol.consumption","Smoking.status","Deep.sleep.percentage","Gender")])
gb_predictions <- predict(gb_model, newdata = test_data[, c("Age","Light.sleep.percentage","Awakenings","Alcohol.consumption","Smoking.status","Deep.sleep.percentage","Gender")])
#----
# Calculate RMSE
dt_rmse <- rmse(dt_predictions, test_data$Sleep.efficiency)
rf_rmse <- rmse(rf_predictions, test_data$Sleep.efficiency)
svm_rmse <- rmse(svm_predictions, test_data$Sleep.efficiency)
gb_rmse <- rmse(gb_predictions, test_data$Sleep.efficiency)

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
#-----
# Load required libraries
library(ggplot2)

# Create a data frame with model names and their corresponding metrics
metrics_df <- data.frame(
  Model = c("Decision Tree", "Random Forest", "SVR", "Gradient Boosting"),
  R_squared = c(dt_r_squared, rf_r_squared, svm_r_squared, gb_r_squared),
  MAE = c(dt_mae, rf_mae, svm_mae, gb_mae),
  RMSE = c(dt_rmse, rf_rmse, svm_rmse, gb_rmse)
  
)

# Melt the data frame to long format for plotting
melted_metrics <- reshape2::melt(metrics_df, id.vars = "Model", variable.name = "Metric", value.name = "Value")

# Create a bar plot
barplot <- ggplot(melted_metrics, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = round(Value, 2)), vjust = -0.5, position = position_dodge(width = 0.9)) +  # Add data labels
  labs(title = "Performance Comparison of Different Models using Lasso Regression",
       y = "Value", fill = "Metric") +
  theme_minimal()

# Display the bar plot
print(barplot)

