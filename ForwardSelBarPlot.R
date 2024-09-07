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

# Forward feature selection
forward_feature_selection <- function(data, target_variable, num_features = 8) {
  # Initialize selected features and remaining features
  selected_features <- character(0)
  remaining_features <- setdiff(names(data), target_variable)
  
  for (i in 1:num_features) {
    p_values <- numeric(length(remaining_features))
    
    # Fit models with each remaining feature
    for (j in seq_along(remaining_features)) {
      model_formula <- as.formula(paste(target_variable, "~", paste(c(selected_features, remaining_features[j]), collapse = "+")))
      temp_model <- lm(model_formula, data = data)
      p_values[j] <- summary(temp_model)$coefficients[2, "Pr(>|t|)"]
    }
    
    # Select feature with lowest p-value
    best_feature <- remaining_features[which.min(p_values)]
    
    # Update selected features and remove selected feature from remaining features
    selected_features <- c(selected_features, best_feature)
    remaining_features <- setdiff(remaining_features, best_feature)
  }
  
  return(selected_features)
}

# Perform forward feature selection
selected_features <- forward_feature_selection(train_data, "Sleep.efficiency", num_features = 8)

# Train models with selected features
dt_model <- rpart(Sleep.efficiency ~ ., data = train_data[, c(selected_features, "Sleep.efficiency")], method = "anova")
rf_model <- randomForest(Sleep.efficiency ~ ., data = train_data[, c(selected_features, "Sleep.efficiency")])
svm_model <- svm(Sleep.efficiency ~ ., data = train_data[, c(selected_features, "Sleep.efficiency")])
gb_model <- gbm(Sleep.efficiency ~ ., data = train_data[, c(selected_features, "Sleep.efficiency")], distribution = "gaussian")

# Evaluate models
dt_predictions <- predict(dt_model, newdata = test_data[, c(selected_features)])
rf_predictions <- predict(rf_model, newdata = test_data[, c(selected_features)])
svm_predictions <- predict(svm_model, newdata = test_data[, c(selected_features)])
gb_predictions <- predict(gb_model, newdata = test_data[, c(selected_features)])

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
#-------------------------
#MAE  
dt_mae <- mae(dt_predictions, test_data$Sleep.efficiency)
rf_mae <- mae(rf_predictions, test_data$Sleep.efficiency)
svm_mae <- mae(svm_predictions, test_data$Sleep.efficiency)
gb_mae <- mae(gb_predictions, test_data$Sleep.efficiency)
#Printing RMAE
cat("\nDecision Tree MAE:", dt_mae, "\n")
cat("Random Forest MAE:", rf_mae, "\n")
cat("SVR MAE:", svm_mae, "\n")
cat("Gradient Boosting MAE:", gb_mae, "\n")

#------------------------
# Calculate Residual Sum of Squares (RSS) for each model
dt_rss <- sum((dt_predictions - test_data$Sleep.efficiency)^2)
rf_rss <- sum((rf_predictions - test_data$Sleep.efficiency)^2)
svm_rss <- sum((svm_predictions - test_data$Sleep.efficiency)^2)
gb_rss <- sum((gb_predictions - test_data$Sleep.efficiency)^2)

# Calculate Total Sum of Squares (TSS)
mean_y <- mean(test_data$Sleep.efficiency)
tss <- sum((test_data$Sleep.efficiency - mean_y)^2)

# Calculate R-squared for each model
dt_r_squared <- 1 - (dt_rss / tss)
rf_r_squared <- 1 - (rf_rss / tss)
svm_r_squared <- 1 - (svm_rss / tss)
gb_r_squared <- 1 - (gb_rss / tss)

# Print R-squared for each model
cat("\nDecision Tree R-squared:", dt_r_squared, "\n")
cat("Random Forest R-squared:", rf_r_squared, "\n")
cat("SVR R-squared:", svm_r_squared, "\n")
cat("Gradient Boosting R-squared:", gb_r_squared, "\n")
##----
# library(caret)
# 
# # Identify numeric columns
# numeric_columns <- sapply(train_data, is.numeric)
# 
# # Identify non-numeric columns
# non_numeric_columns <- names(train_data)[!numeric_columns]
# 
# # Encode Gender column into two separate binary columns
# encoded_gender <- model.matrix(~ Gender - 1, data = train_data)
# 
# # Encode Smoking.status column into two separate binary columns
# encoded_smoking <- model.matrix(~ Smoking.status - 1, data = train_data)
# 
# # Combine encoded columns with numeric columns
# combined_train_data <- cbind(train_data[, numeric_columns], encoded_gender, encoded_smoking)
# 
# # Compute correlation matrix
# correlation_matrix <- cor(combined_train_data)
# 
# # Find correlated features
# correlated_features <- findCorrelation(correlation_matrix, cutoff = 0.8)
# 
# # Identify important features by removing correlated ones
# important_features <- names(combined_train_data)[-correlated_features]
# 
# # Extract correlation coefficients for important features
# correlation_scores <- correlation_matrix[rownames(correlation_matrix) %in% important_features, important_features]
# 
# # Print column names and their corresponding correlation coefficients
# print("Important Features and Their Correlation Scores:")
# print(correlation_scores)
# 

##----
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
  labs(title = "Performance Comparison of Different Models using Forward FS",
       y = "Value", fill = "Metric") +
  theme_minimal()

# Display the bar plot
print(barplot)
