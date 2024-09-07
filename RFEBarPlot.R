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

# Print importance scores
print(importance_scores)


# Sort importance scores in descending order
sorted_importance <- importance_scores[order(importance_scores, decreasing = TRUE), ]

# Print importance scores
print(as.list(sorted_importance))
selected_col<- names(sorted_importance)[1:8]

#-------
# Decision Tree Model
dt_model <- rpart(Sleep.efficiency ~ ., data = train_data[, c(selected_col, "Sleep.efficiency")], method = "anova")

# Random Forest Model
rf_model <- randomForest(Sleep.efficiency ~ ., data = train_data[, c(selected_col, "Sleep.efficiency")])

# Support Vector Regression (SVR) Model
svm_model <- svm(Sleep.efficiency ~ ., data = train_data[, c(selected_col, "Sleep.efficiency")])

# Gradient Boosting Model
gb_model <- gbm(Sleep.efficiency ~ ., data = train_data[, c(selected_col, "Sleep.efficiency")], distribution = "gaussian")

# Evaluate models
dt_predictions <- predict(dt_model, newdata = test_data[, selected_col])
rf_predictions <- predict(rf_model, newdata = test_data[, selected_col])
svm_predictions <- predict(svm_model, newdata = test_data[, selected_col])
gb_predictions <- predict(gb_model, newdata = test_data[, selected_col])

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
#----
# Create a data frame for bar graph
barplot_data <- data.frame(
  Model = c("Decision Tree", "Random Forest", "SVR", "Gradient Boosting"),
  R_squared = c(dt_r_squared, rf_r_squared, svm_r_squared, gb_r_squared),
  MAE = c(dt_mae, rf_mae, svm_mae, gb_mae),
  RMSE = c(dt_rmse, rf_rmse, svm_rmse, gb_rmse)
)

# Melt the data frame for easier plotting
library(reshape2)
melted_data <- melt(barplot_data, id.vars = "Model")

# Create the bar plot
library(ggplot2)
# Create the bar plot with data labels
barplot <- ggplot(melted_data, aes(x = Model, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = round(value, 2)), position = position_dodge(width = 0.9), vjust = -0.5) +  # Add data labels
  labs(title = "Performance Comparison of Different Models using RFE",
       y = "Value", fill = "Metric") +
  theme_minimal()

# Save the bar plot as an image
ggsave("barplot_with_labels.png", plot = barplot, width = 8, height = 6, units = "in", dpi = 300)

# Display the bar plot
print(barplot)
